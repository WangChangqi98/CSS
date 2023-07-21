import copy
import torch
import torch.nn as nn
from generalframeworks.networks.deeplabv3.deeplabv3 import DeepLabv3Plus_with_rep
import torch.nn.functional as F
from generalframeworks.dataset_helpers.VOC import batch_transform, generate_cut_gather, batch_transform_2, batch_transform_3, generate_cut_gather_2, generate_cut_gather_3

class Model_ori_pseudo(nn.Module):
    '''
    Build a model for DDP with: a DeepLabV3_Plus, a ema, and a mlp
    '''

    def __init__(self, base_encoder, num_classes=21, output_dim=256, ema_alpha=0.99, config=None) -> None:
        super(Model_ori_pseudo, self).__init__()
        self.model = DeepLabv3Plus_with_rep(base_encoder, num_classes=num_classes, output_dim=output_dim, dilate_scale=8)
        ##### Init EMA #####
        self.step = 0
        self.ema_model = copy.deepcopy(self.model)
        for p in self.ema_model.parameters():
            p.requires_grad = False
        self.alpha = ema_alpha
        print('EMA model has been prepared. Alpha = {}'.format(self.alpha))

        self.config = config

    def ema_update(self):
        decay = min(1 - 1 / (self.step + 1), self.alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data
        self.step += 1

    def forward(self, train_l_image, train_u_image):
        ##### generate pseudo label #####
        with torch.no_grad():
            pred_u, _ = self.ema_model(train_u_image)
            pred_u_large_raw = F.interpolate(pred_u, size=train_u_image.shape[2:], mode='bilinear', align_corners=True)
            pseudo_logits, pseudo_labels = torch.max(torch.softmax(pred_u_large_raw, dim=1), dim=1)

            # Randomly scale images
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = batch_transform(train_u_image, pseudo_labels,
                                                                                       pseudo_logits,
                                                                                       crop_size=self.config['Dataset']['crop_size'],
                                                                                       scale_size=self.config['Dataset']['scale_size'],
                                                                                       augmentation=False)
            # Apply mixing strategy, we gather all images cross mutiple GPUs during this progress
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = generate_cut_gather(train_u_aug_image,
                                                                                    train_u_aug_label,
                                                                                    train_u_aug_logits,
                                                                                    mode=self.config['Dataset'][
                                                                                        'mix_mode'])
            # Apply augmnetation : color jitter + flip + gaussian blur
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = batch_transform(train_u_aug_image,
                                                                                       train_u_aug_label,
                                                                                       train_u_aug_logits,
                                                                                       crop_size=self.config['Dataset']['crop_size'],
                                                                                       scale_size=(1.0, 1.0),
                                                                                       augmentation=True)


        pred_l, rep_l = self.model(train_l_image)
        pred_l_large = F.interpolate(pred_l, size=train_l_image.shape[2:], mode='bilinear', align_corners=True)

        pred_u, rep_u = self.model(train_u_aug_image)
        pred_u_large = F.interpolate(pred_u, size=train_l_image.shape[2:], mode='bilinear', align_corners=True)


        rep_all = torch.cat((rep_l, rep_u))
        pred_all = torch.cat((pred_l, pred_u))

        return pred_l_large, pred_u_large, train_u_aug_label, train_u_aug_logits, rep_all, pred_all, pred_u_large_raw


class Model_mix(nn.Module):
    '''
    Build a model for DDP with: a DeepLabV3_Plus, a ema, and a mlp
    '''

    def __init__(self, base_encoder, num_classes=21, output_dim=256, ema_alpha=0.99, config=None, temp=0.25) -> None:
        super(Model_mix, self).__init__()
        self.model = DeepLabv3Plus_with_rep(base_encoder, num_classes=num_classes, output_dim=output_dim, dilate_scale=8)
        self.temp = temp
        self.num_classes = num_classes
        ##### Init EMA #####
        self.step = 0
        self.ema_model = copy.deepcopy(self.model)
        for p in self.ema_model.parameters():
            p.requires_grad = False
        self.alpha = ema_alpha
        print('EMA model has been prepared. Alpha = {}'.format(self.alpha))

        self.config = config

    def ema_update(self):
        decay = min(1 - 1 / (self.step + 1), self.alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data
        self.step += 1

    def forward(self, train_l_image, train_u_image, prototypes):
        ##### generate pseudo label from class predictor and indicator from representation predictor #####
        with torch.no_grad():
            pred_l, rep_l = self.ema_model(train_l_image)
            pred_u, rep_u = self.ema_model(train_u_image)
            rep_b, rep_dim, rep_w, rep_h = rep_u.shape
            norm_rep_u = rep_u.permute(0, 2, 3, 1)
            norm_rep_u = F.normalize(norm_rep_u, dim=-1)
            norm_proto = F.normalize(prototypes, dim=-1).permute(1, 0)
            norm_rep_u = norm_rep_u.reshape(rep_b * rep_w * rep_h, rep_dim)
            sim_mat = torch.mm(norm_rep_u, norm_proto)
            sim_mat = sim_mat.reshape(rep_b, rep_w, rep_h, self.num_classes).permute(0, 3, 1, 2)
            sim_mat_large_raw = F.interpolate(sim_mat, size=train_u_image.shape[2:], mode='bilinear', align_corners=True)
            pseudo_logits_rep, pseudo_labels_rep = torch.max(F.softmax(sim_mat_large_raw / self.temp, dim=1), dim=1)
            pred_u_large_raw = F.interpolate(pred_u, size=train_u_image.shape[2:], mode='bilinear', align_corners=True)
            pseudo_logits_cls, pseudo_labels_cls = torch.max(torch.softmax(pred_u_large_raw, dim=1), dim=1)
            label_mask = pseudo_labels_cls.eq(pseudo_labels_rep)
            label_mask = (~label_mask).float()
            pseudo_labels = pseudo_labels_cls - label_mask * self.num_classes
            pseudo_labels[pseudo_labels < 0] = 255

            # Randomly scale images
            train_u_aug_image, train_u_aug_label, train_u_aug_logits_cls, train_u_aug_logits_rep = batch_transform_2(train_u_image, pseudo_labels,
                                                                                                                    pseudo_logits_cls, pseudo_logits_rep,
                                                                                                                    crop_size=self.config['Dataset']['crop_size'],
                                                                                                                    scale_size=self.config['Dataset']['scale_size'],
                                                                                                                    augmentation=False)
            # Apply mixing strategy, we gather all images cross mutiple GPUs during this progress
            train_u_aug_image, train_u_aug_label, train_u_aug_logits_cls, train_u_aug_logits_rep = generate_cut_gather_2(train_u_aug_image,
                                                                                                                         train_u_aug_label,
                                                                                                                         train_u_aug_logits_cls, train_u_aug_logits_rep,
                                                                                                                         mode=self.config['Dataset']['mix_mode'])
            # Apply augmnetation : color jitter + flip + gaussian blur
            train_u_aug_image, train_u_aug_label, train_u_aug_logits_cls, train_u_aug_logits_rep = batch_transform_2(train_u_aug_image,
                                                                                                                     train_u_aug_label,
                                                                                                                     train_u_aug_logits_cls, train_u_aug_logits_rep,
                                                                                                                     crop_size=self.config['Dataset']['crop_size'],
                                                                                                                     scale_size=(1.0, 1.0),
                                                                                                                     augmentation=True)


        pred_l, rep_l = self.model(train_l_image)
        pred_l_large = F.interpolate(pred_l, size=train_l_image.shape[2:], mode='bilinear', align_corners=True)

        pred_u, rep_u = self.model(train_u_aug_image)
        pred_u_large = F.interpolate(pred_u, size=train_l_image.shape[2:], mode='bilinear', align_corners=True)


        rep_all = torch.cat((rep_l, rep_u))
        rep_all_b = rep_all.shape[0]
        norm_rep_all = rep_all.permute(0, 2, 3, 1)
        norm_rep_all = F.normalize(norm_rep_all, dim=-1)
        norm_rep_all = norm_rep_all.reshape(rep_all_b * rep_w * rep_h, rep_dim)
        prob_all = torch.mm(norm_rep_all, norm_proto)
        prob_all = prob_all.reshape(rep_all_b, rep_w, rep_h, self.num_classes).permute(0, 3, 1, 2)
        prob_all = F.softmax(prob_all / self.temp, dim=1)

        return pred_l_large, pred_u_large, train_u_aug_label, train_u_aug_logits_cls, train_u_aug_logits_rep, rep_all, prob_all

class Model_cross(nn.Module):
    '''
    Build a model for DDP with: a DeepLabV3_Plus, a ema, and a mlp
    '''

    def __init__(self, base_encoder, num_classes=21, output_dim=256, ema_alpha=0.99, config=None, temp=0.1) -> None:
        super(Model_cross, self).__init__()
        self.model = DeepLabv3Plus_with_rep(base_encoder, num_classes=num_classes, output_dim=output_dim, dilate_scale=8)
        self.temp = temp
        self.num_classes = num_classes
        ##### Init EMA #####
        self.step = 0
        self.ema_model = copy.deepcopy(self.model)
        for p in self.ema_model.parameters():
            p.requires_grad = False
        self.alpha = ema_alpha
        print('EMA model has been prepared. Alpha = {}'.format(self.alpha))

        self.config = config

    def ema_update(self):
        decay = min(1 - 1 / (self.step + 1), self.alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data
        self.step += 1

    def forward(self, train_l_image, train_u_image, prototypes):
        ##### generate pseudo label from class predictor and indicator from representation predictor #####
        with torch.no_grad():
            pred_l, rep_l = self.ema_model(train_l_image)
            pred_u, rep_u = self.ema_model(train_u_image)
            rep_b, rep_dim, rep_w, rep_h = rep_u.shape
            norm_rep_u = rep_u.permute(0, 2, 3, 1)
            norm_rep_u = F.normalize(norm_rep_u, dim=-1)
            norm_proto = F.normalize(prototypes, dim=-1).permute(1, 0)
            norm_rep_u = norm_rep_u.reshape(rep_b * rep_w * rep_h, rep_dim)
            sim_mat = torch.mm(norm_rep_u, norm_proto)
            sim_mat = sim_mat.reshape(rep_b, rep_w, rep_h, self.num_classes).permute(0, 3, 1, 2)
            sim_mat_large_raw = F.interpolate(sim_mat, size=train_u_image.shape[2:], mode='bilinear', align_corners=True)
            pseudo_logits_rep, pseudo_labels_rep = torch.max(F.softmax(sim_mat_large_raw / self.temp, dim=1), dim=1)
            pred_u_large_raw = F.interpolate(pred_u, size=train_u_image.shape[2:], mode='bilinear', align_corners=True)
            pseudo_logits_cls, pseudo_labels_cls = torch.max(torch.softmax(pred_u_large_raw, dim=1), dim=1)

            # Randomly scale images
            train_u_aug_image, train_u_aug_label_cls, train_u_aug_label_rep, train_u_aug_logits_cls, train_u_aug_logits_rep = batch_transform_3(train_u_image, pseudo_labels_cls, pseudo_labels_rep,
                                                                                                                    pseudo_logits_cls, pseudo_logits_rep,
                                                                                                                    crop_size=self.config['Dataset']['crop_size'],
                                                                                                                    scale_size=self.config['Dataset']['scale_size'],
                                                                                                                    augmentation=False)
            # Apply mixing strategy, we gather all images cross mutiple GPUs during this progress
            train_u_aug_image, train_u_aug_label_cls, train_u_aug_label_rep, train_u_aug_logits_cls, train_u_aug_logits_rep = generate_cut_gather_3(train_u_aug_image,
                                                                                                                         train_u_aug_label_cls,
                                                                                                                         train_u_aug_label_rep,
                                                                                                                         train_u_aug_logits_cls, train_u_aug_logits_rep,
                                                                                                                         mode=self.config['Dataset']['mix_mode'])
            # Apply augmnetation : color jitter + flip + gaussian blur
            train_u_aug_image, train_u_aug_label_cls, train_u_aug_label_rep, train_u_aug_logits_cls, train_u_aug_logits_rep = batch_transform_3(train_u_aug_image,
                                                                                                                     train_u_aug_label_cls,
                                                                                                                     train_u_aug_label_rep,
                                                                                                                     train_u_aug_logits_cls, train_u_aug_logits_rep,
                                                                                                                     crop_size=self.config['Dataset']['crop_size'],
                                                                                                                     scale_size=(1.0, 1.0),
                                                                                                                     augmentation=True)


        pred_l, rep_l = self.model(train_l_image)
        pred_l_large = F.interpolate(pred_l, size=train_l_image.shape[2:], mode='bilinear', align_corners=True)

        pred_u, rep_u = self.model(train_u_aug_image)
        pred_u_large = F.interpolate(pred_u, size=train_l_image.shape[2:], mode='bilinear', align_corners=True)


        rep_all = torch.cat((rep_l, rep_u))
        rep_all_b = rep_all.shape[0]
        norm_rep_all = rep_all.permute(0, 2, 3, 1)
        norm_rep_all = F.normalize(norm_rep_all, dim=-1)
        norm_rep_all = norm_rep_all.reshape(rep_all_b * rep_w * rep_h, rep_dim)
        prob_all = torch.mm(norm_rep_all, norm_proto)
        prob_all = prob_all.reshape(rep_all_b, rep_w, rep_h, self.num_classes).permute(0, 3, 1, 2)
        prob_all = F.softmax(prob_all / self.temp, dim=1)

        return pred_l_large, pred_u_large, train_u_aug_label_cls, train_u_aug_label_rep, train_u_aug_logits_cls, train_u_aug_logits_rep, rep_all, prob_all

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    Warning: torch.distributed.all_ather has no gradient.
    """
    tensor_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensor_gather, tensor, async_op=False)
    output = torch.cat(tensor_gather, dim=0)

    return output