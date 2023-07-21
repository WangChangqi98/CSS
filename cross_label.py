import shutup
shutup.please()
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
from generalframeworks.dataset_helpers.VOC import VOC_BuildData
from generalframeworks.dataset_helpers.Cityscapes import City_BuildData
from generalframeworks.networks.ddp_model import Model_cross
from generalframeworks.scheduler.my_lr_scheduler import PolyLR
from generalframeworks.scheduler.rampscheduler import RampdownScheduler
from generalframeworks.utils import iterator_, Logger
from generalframeworks.util.meter import *
from generalframeworks.utils import label_onehot
from generalframeworks.util.torch_dist_sum import *
from generalframeworks.util.miou import *
from generalframeworks.util.dist_init import local_dist_init
from generalframeworks.loss.loss import ProbOhemCrossEntropy2d, Attention_Threshold_Loss, Contrast_Loss
import yaml
import os
import time
import torchvision.models as models
import argparse
import random

def main(rank, config, args):
    ##### Distribution init #####
    local_dist_init(args, rank)
    print('Hello from rank {}\n'.format(rank))

    ##### Load the dataset #####
    if config['Dataset']['name'] == 'VOC':
        data = VOC_BuildData(data_path=config['Dataset']['data_dir'], txt_path=config['Dataset']['txt_dir'], 
                          label_num=args.num_labels, seed=config['Seed'], crop_size=config['Dataset']['crop_size'])
    if config['Dataset']['name'] == 'CityScapes':
        data = City_BuildData(data_path=config['Dataset']['data_dir'], txt_path=config['Dataset']['txt_dir'], 
                          label_num=args.num_labels, seed=config['Seed'], crop_size=config['Dataset']['crop_size'])
    train_l_dataset, train_u_dataset, test_dataset = data.build()
    train_l_sampler = torch.utils.data.distributed.DistributedSampler(train_l_dataset)
    train_l_loader = torch.utils.data.DataLoader(train_l_dataset, 
                                                 batch_size=config['Dataset']['batch_size'],
                                                 pin_memory=True,
                                                 sampler=train_l_sampler,
                                                 num_workers=4,
                                                 drop_last=True)
    train_u_sampler = torch.utils.data.distributed.DistributedSampler(train_u_dataset)
    train_u_loader = torch.utils.data.DataLoader(train_u_dataset, 
                                                 batch_size=config['Dataset']['batch_size'],
                                                 pin_memory=True,
                                                 sampler=train_u_sampler,
                                                 num_workers=4,
                                                 drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                              batch_size=config['Dataset']['batch_size'],
                                              pin_memory=True,
                                              num_workers=4)

    ##### Load the weight for each class #####
    weight = torch.FloatTensor(
    [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
        0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
        1.0865, 1.1529, 1.0507]).cuda()

    ##### Model init #####
    backbone = models.resnet101()
    ckpt = torch.load('./pretrained/resnet101.pth', map_location='cpu')
    backbone.load_state_dict(ckpt)
    
    # for Resnet-101 stem users
    #backbone = resnet.resnet101(pretrained=True)

    model = Model_cross(backbone, num_classes=config['Network']['num_class'], output_dim=256, config=config, temp=args.temp).cuda()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)

    ##### Loss init #####
    criterion = {'sup_loss': ProbOhemCrossEntropy2d(ignore_label=-1, thresh=0.7, min_kept=50000 * config['Dataset']['batch_size']).cuda(),
                 'ce_loss': nn.CrossEntropyLoss(ignore_index=-1).cuda(),
                 'unsup_loss': Attention_Threshold_Loss(strong_threshold=args.un_threshold).cuda(),
                 'contrast_loss': Contrast_Loss(strong_threshold=args.strong_threshold, 
                                                num_queries=config['Loss']['num_queries'],
                                                num_negatives=config['Loss']['num_negatives'],
                                                temp=config['Loss']['temp'],
                                                alpha=config['Loss']['alpha']).cuda(),
                }

    ##### Prototype init #####
    global prototypes

    prototypes = torch.zeros(config['Network']['num_class'], 256).cuda()
    if os.path.exists(args.prototypes_resume):
        print('prototypes resume from', args.prototypes_resume)
        checkpoint = torch.load(args.prototypes_resume, map_location='cpu')
        prototypes = torch.tensor(checkpoint['prototypes']).cuda()

    ##### Other init #####
    optimizer = torch.optim.SGD(model.module.model.parameters(),
                                 lr=float(config['Optim']['lr']), weight_decay=float(config['Optim']['weight_decay']), momentum=0.9, nesterov=True)
    total_iter = args.total_iter
    total_epoch = int(total_iter / len(train_l_loader))
    if dist.get_rank() == 0:
        print('total epoch is {}'.format(total_epoch))
    lr_scheduler = PolyLR(optimizer, total_iter, min_lr=1e-4)

    if os.path.exists(args.resume):
        print('resume from', args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.module.model.load_state_dict(checkpoint['model'])
        model.module.ema_model.load_state_dict(checkpoint['ema_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        prototypes = torch.tensor(checkpoint['prototypes']).cuda()
    else:
        start_epoch = 0
    sche_d = RampdownScheduler(begin_epoch=config['Ramp_Scheduler']['begin_epoch'], 
                        max_epoch=config['Ramp_Scheduler']['max_epoch'],
                        current_epoch=start_epoch,
                        max_value=config['Ramp_Scheduler']['max_value'],
                        min_value=config['Ramp_Scheduler']['min_value'], 
                        ramp_mult=config['Ramp_Scheduler']['ramp_mult'])

    # if dist.get_rank() == 0:
    #     log = Logger(logFile='./log/' + str(args.job_name) + '.log')
    best_miou = 0

    model.module.model.train()
    model.module.ema_model.train()
    for epoch in range(start_epoch, total_epoch):
        train(train_l_loader, train_u_loader, model, optimizer, criterion, epoch, lr_scheduler, sche_d, config, args)
        miou = test(test_loader, model.module.ema_model, config)
        best_miou = max(best_miou, miou)
        if dist.get_rank() == 0:
            print('Epoch:{} * mIoU {:.4f} Best_mIoU {:.4f} Time {}'.format(epoch, miou, best_miou, time.asctime(time.localtime(time.time()))))
            # log.write('Epoch:{} * mIoU {:.4f} Best_mIoU {:.4f} Time {}\n'.format(epoch, miou, best_miou, time.asctime( time.localtime(time.time()) )))
            # Save model
            if miou == best_miou:
                save_dir = './checkpoints/' + str(args.job_name)
                torch.save(
                    {
                        'epoch': epoch+1,
                        'model': model.module.model.state_dict(),
                        'ema_model': model.module.ema_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'prototypes': prototypes.data.cpu().numpy(),
                    }, os.path.join(save_dir, 'best_model.pth'))
        


def train(train_l_loader, train_u_loader, model, optimizer, criterion, epoch, scheduler, sche_d, config, args):
    num_class = config['Network']['num_class'] 
    # switch to train mode
    model.module.model.train()
    model.module.ema_model.train()

    train_u_loader.sampler.set_epoch(epoch)
    training_u_iter = iterator_(train_u_loader)
    train_l_loader.sampler.set_epoch(epoch)
    for i, (train_l_image, train_l_label) in enumerate(train_l_loader):
        train_l_image, train_l_label = train_l_image.cuda(), train_l_label.cuda()
        train_u_image, train_u_label = training_u_iter.__next__()
        train_u_image, train_u_label = train_u_image.cuda(), train_u_label.cuda()
        pred_l_large, pred_u_large, train_u_aug_label_cls, train_u_aug_label_rep, train_u_aug_logits_cls, train_u_aug_logits_rep, rep_all, pred_all = model(train_l_image, train_u_image, prototypes)

        if config['Dataset']['name'] == 'VOC':
            sup_loss = criterion['ce_loss'](pred_l_large, train_l_label)
        else:
            sup_loss = criterion['sup_loss'](pred_l_large, train_l_label)
        if epoch < args.warmup:
            unsup_loss = criterion['unsup_loss'](pred_u_large, train_u_aug_label_cls, train_u_aug_logits_cls)
        else:
            unsup_loss = criterion['unsup_loss'](pred_u_large, train_u_aug_label_rep, train_u_aug_logits_rep)

        ##### Contrastive learning #####
        with torch.no_grad():
            train_u_aug_mask = train_u_aug_logits_cls.ge(args.weak_threshold).float()
            mask_all = torch.cat(((train_l_label.unsqueeze(1) >= 0).float(), train_u_aug_mask.unsqueeze(1)))
            mask_all = F.interpolate(mask_all, size=pred_all.shape[2:], mode='nearest')

            label_l = F.interpolate(label_onehot(train_l_label, num_class), size=pred_all.shape[2:], mode='nearest')
            label_u = F.interpolate(label_onehot(train_u_aug_label_cls, num_class), size=pred_all.shape[2:], mode='nearest')
            label_all = torch.cat((label_l, label_u))

        contrast_loss = criterion['contrast_loss'](rep_all, label_all, mask_all, pred_all, prototypes)

        if args.sche:
            total_loss = sup_loss + unsup_loss + contrast_loss * sche_d.value
        else:
            total_loss = sup_loss + unsup_loss + contrast_loss

        # Update Meter
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        model.module.ema_update()
        scheduler.step()
    sche_d.step()

@torch.no_grad()
def test(test_loader, model, config):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    miou_meter = ConfMatrix(num_classes=config['Network']['num_class'], fmt=':6.4f', name='test_miou')

    # switch to eval mode
    model.eval()

    end = time.time()
    test_iter = iter(test_loader)
    for _ in range(len(test_loader)):
        data_time.update(time.time() - end)
        test_image, test_label = test_iter.next()
        test_image, test_label = test_image.cuda(), test_label.cuda()
        
        pred, _ = model(test_image)
        pred = F.interpolate(pred, size=test_label.shape[1:], mode='bilinear', align_corners=True)

        miou_meter.update(pred.argmax(1).flatten(), test_label.flatten())
        batch_time.update(time.time() - end)
        end = time.time()

    mat = torch_dist_sum(dist.get_rank(), miou_meter.mat)
    miou = mean_intersection_over_union(mat[0]) 

    return miou


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/VOC_config_baseline.yaml')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--prototypes_resume', type=str, default='')
    parser.add_argument('--num_labels', type=int, default=92)
    parser.add_argument('--job_name', type=str, default='VOC_92_cross_label')

    # Distributed
    parser.add_argument('--gpu_id', type=str, default='0,1,2,3')
    parser.add_argument('--world_size', type=str, default='4')
    parser.add_argument('--port', type=str, default='12301')

    # Hyperparameter
    parser.add_argument('--strong_threshold', type=float, default=0.8)
    parser.add_argument('--weak_threshold', type=float, default=0.7)
    parser.add_argument('--un_threshold', type=float, default=0.97)
    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--sche', type=bool, default=True)

    args = parser.parse_args()

    ##### Config init #####
    with open(args.config, 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    save_dir = './checkpoints/' + str(args.job_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + '/config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(config)

    ##### Init Seed #####
    random.seed(config['Seed'])
    torch.manual_seed(config['Seed'])
    torch.backends.cudnn.deterministic = True

    mp.spawn(main, nprocs=int(args.world_size), args=(config, args))
