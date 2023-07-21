import torch.utils.data as data
import torch
import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
import random
from PIL import Image, ImageFilter
import numpy as np
import torch.distributed as dist

class Pascal_VOC_Dataset(data.Dataset):
    def __init__(self, root, idx_list, crop_size=(512, 512), scale_size=(0.5, 2.0), augmentation=True, train=True):
        self.root = os.path.expanduser(root)
        self.train = train
        self.crop_size = crop_size
        self.augmentation = augmentation
        self.scale_size = scale_size
        self.idx_list = idx_list
        
    def __getitem__(self, index):
        image_root = Image.open(self.root + '/JPEGImages/{}.jpg'.format(self.idx_list[index]))
        label_root = Image.open(self.root + '/SegmentationClassAug/{}.png'.format(self.idx_list[index]))
        image, label = transform(image_root, label_root, None, crop_size=self.crop_size, scale_size=self.scale_size, augmentation=self.augmentation)
        return image, label.squeeze(0)
    
    def __len__(self):
        return len(self.idx_list)
            
class VOC_BuildData():
    def __init__(self, data_path, txt_path, label_num, seed, crop_size=[512,512]):
        self.data_path = data_path
        self.txt_path = txt_path
        self.image_size = [513, 513]
        self.crop_size = crop_size
        self.num_segments = 21
        self.scale_size = (0.5, 1.5)
        self.train_l_idx, self.train_u_idx, self.test_idx= get_pascal_idx_via_txt(self.txt_path, label_num=label_num, seed=seed)
        
    def build(self):
        train_l_dataset = Pascal_VOC_Dataset(self.data_path, self.train_l_idx, self.crop_size, self.scale_size,
                                             augmentation=True, train=True)
        train_u_dataset = Pascal_VOC_Dataset(self.data_path, self.train_u_idx, self.crop_size, scale_size=(1.0, 1.0),
                                                augmentation=False, train=True)
        test_dataset = Pascal_VOC_Dataset(self.data_path, self.test_idx, self.crop_size, scale_size=(1.0, 1.0),augmentation=False,
                                          train=False)
        return train_l_dataset, train_u_dataset, test_dataset

def get_pascal_idx_via_txt(root, label_num, seed):
    '''
    Read idx list via generated txt, pre-perform make_list.py
    '''
    root = root + '/' + str(label_num) + '/' + str(seed)
    with open(root + '/labeled_filename.txt') as f:
        labeled_list = f.read().splitlines()
    f.close()
    with open(root + '/unlabeled_filename.txt') as f:
        unlabeled_list = f.read().splitlines()
    f.close()
    with open(root + '/valid_filename.txt') as f:
        test_list = f.read().splitlines()
    f.close()
    return labeled_list, unlabeled_list, test_list

def transform(image, label, logits=None, crop_size=(512, 512), scale_size=(0.8, 1.0), augmentation=True):
    # Randomly rescale images
    raw_w, raw_h = image.size
    scale_ratio = random.uniform(scale_size[0], scale_size[1])

    resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
    image = transforms_f.resize(image, resized_size, Image.BILINEAR)
    label = transforms_f.resize(label, resized_size, Image.NEAREST)
    if logits is not None:
        logits = transforms_f.resize(logits, resized_size, Image.NEAREST)

    # Add padding if rescaled image is smaller than crop size
    if crop_size == -1: # Use original image size
        crop_size = (raw_w, raw_h)

    if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
        right_pad = max(crop_size[1] - resized_size[1], 0)
        bottom_pad = max(crop_size[0] - resized_size[0], 0)
        image = transforms_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')
        label = transforms_f.pad(label, padding=(0, 0, right_pad, bottom_pad), fill=255, padding_mode='constant')
        if logits is not None:
            logits = transforms_f.pad(logits, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')
    
    # Randomly crop images
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = transforms_f.crop(image, i, j, h, w)
    label = transforms_f.crop(label, i, j, h, w)
    if logits is not None:
        logits = transforms_f.crop(logits, i, j, h, w)
    
    if augmentation:
        # Random color jittering
        if torch.rand(1) > 0.2:
            color_transform = transforms.ColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))
            image = color_transform(image)
        
        # Random Gaussian filtering
        if torch.rand(1) > 0.5:
            sigma = random.uniform(0.15, 1.15)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        # Random horizontal flipping
        if torch.rand(1) > 0.5:
            image = transforms_f.hflip(image)
            label = transforms_f.hflip(label)
            if logits is not None:
                logits = transforms_f.hflip(logits)
        
    # Transform to Tensor
    image = transforms_f.to_tensor(image)
    label = (transforms_f.to_tensor(label) * 255).long()
    label[label == 255] = -1 # invalid pixels are re-mapped to index -1
    if logits is not None:
        logits = transforms_f.to_tensor(logits)

    # Apply ImageNet normalization
    image = transforms_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if logits is not None:
        return image, label, logits
    else:
        return image, label

def transform_2(image, label, logits1=None, logits2=None, crop_size=(512, 512), scale_size=(0.8, 1.0), augmentation=True):
    # Randomly rescale images
    raw_w, raw_h = image.size
    scale_ratio = random.uniform(scale_size[0], scale_size[1])

    resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
    image = transforms_f.resize(image, resized_size, Image.BILINEAR)
    label = transforms_f.resize(label, resized_size, Image.NEAREST)
    if logits1 is not None:
        logits1 = transforms_f.resize(logits1, resized_size, Image.NEAREST)
    if logits2 is not None:
        logits2 = transforms_f.resize(logits2, resized_size, Image.NEAREST)

    # Add padding if rescaled image is smaller than crop size
    if crop_size == -1: # Use original image size
        crop_size = (raw_w, raw_h)

    if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
        right_pad = max(crop_size[1] - resized_size[1], 0)
        bottom_pad = max(crop_size[0] - resized_size[0], 0)
        image = transforms_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')
        label = transforms_f.pad(label, padding=(0, 0, right_pad, bottom_pad), fill=255, padding_mode='constant')
        if logits1 is not None:
            logits1 = transforms_f.pad(logits1, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')
        if logits2 is not None:
            logits2 = transforms_f.pad(logits2, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')
    
    # Randomly crop images
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = transforms_f.crop(image, i, j, h, w)
    label = transforms_f.crop(label, i, j, h, w)
    if logits1 is not None:
        logits1 = transforms_f.crop(logits1, i, j, h, w)
    if logits2 is not None:
        logits2 = transforms_f.crop(logits2, i, j, h, w)
    
    if augmentation:
        # Random color jittering
        if torch.rand(1) > 0.2:
            color_transform = transforms.ColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))
            image = color_transform(image)
        
        # Random Gaussian filtering
        if torch.rand(1) > 0.5:
            sigma = random.uniform(0.15, 1.15)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        # Random horizontal flipping
        if torch.rand(1) > 0.5:
            image = transforms_f.hflip(image)
            label = transforms_f.hflip(label)
            if logits1 is not None:
                logits1 = transforms_f.hflip(logits1)
            if logits2 is not None:
                logits2 = transforms_f.hflip(logits2)
        
    # Transform to Tensor
    image = transforms_f.to_tensor(image)
    label = (transforms_f.to_tensor(label) * 255).long()
    label[label == 255] = -1 # invalid pixels are re-mapped to index -1
    if logits1 is not None:
        logits1 = transforms_f.to_tensor(logits1)
    if logits2 is not None:
        logits2 = transforms_f.to_tensor(logits2)

    # Apply ImageNet normalization
    image = transforms_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if logits1 is not None:
        return image, label, logits1, logits2
    else:
        return image, label

def transform_3(image, label1, label2, logits1=None, logits2=None, crop_size=(512, 512), scale_size=(0.8, 1.0), augmentation=True):
    # Randomly rescale images
    raw_w, raw_h = image.size
    scale_ratio = random.uniform(scale_size[0], scale_size[1])

    resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
    image = transforms_f.resize(image, resized_size, Image.BILINEAR)
    label1 = transforms_f.resize(label1, resized_size, Image.NEAREST)
    label2 = transforms_f.resize(label2, resized_size, Image.NEAREST)
    if logits1 is not None:
        logits1 = transforms_f.resize(logits1, resized_size, Image.NEAREST)
    if logits2 is not None:
        logits2 = transforms_f.resize(logits2, resized_size, Image.NEAREST)

    # Add padding if rescaled image is smaller than crop size
    if crop_size == -1: # Use original image size
        crop_size = (raw_w, raw_h)

    if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
        right_pad = max(crop_size[1] - resized_size[1], 0)
        bottom_pad = max(crop_size[0] - resized_size[0], 0)
        image = transforms_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')
        label1 = transforms_f.pad(label1, padding=(0, 0, right_pad, bottom_pad), fill=255, padding_mode='constant')
        label2 = transforms_f.pad(label2, padding=(0, 0, right_pad, bottom_pad), fill=255, padding_mode='constant')
        if logits1 is not None:
            logits1 = transforms_f.pad(logits1, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')
        if logits2 is not None:
            logits2 = transforms_f.pad(logits2, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')
    
    # Randomly crop images
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = transforms_f.crop(image, i, j, h, w)
    label1 = transforms_f.crop(label1, i, j, h, w)
    label2 = transforms_f.crop(label2, i, j, h, w)
    if logits1 is not None:
        logits1 = transforms_f.crop(logits1, i, j, h, w)
    if logits2 is not None:
        logits2 = transforms_f.crop(logits2, i, j, h, w)
    
    if augmentation:
        # Random color jittering
        if torch.rand(1) > 0.2:
            color_transform = transforms.ColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))
            image = color_transform(image)
        
        # Random Gaussian filtering
        if torch.rand(1) > 0.5:
            sigma = random.uniform(0.15, 1.15)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        # Random horizontal flipping
        if torch.rand(1) > 0.5:
            image = transforms_f.hflip(image)
            label1 = transforms_f.hflip(label1)
            label2 = transforms_f.hflip(label2)
            if logits1 is not None:
                logits1 = transforms_f.hflip(logits1)
            if logits2 is not None:
                logits2 = transforms_f.hflip(logits2)
        
    # Transform to Tensor
    image = transforms_f.to_tensor(image)
    label1 = (transforms_f.to_tensor(label1) * 255).long()
    label2 = (transforms_f.to_tensor(label2) * 255).long()
    label1[label1 == 255] = -1 # invalid pixels are re-mapped to index -1
    label2[label2 == 255] = -1 # invalid pixels are re-mapped to index -1
    if logits1 is not None:
        logits1 = transforms_f.to_tensor(logits1)
    if logits2 is not None:
        logits2 = transforms_f.to_tensor(logits2)

    # Apply ImageNet normalization
    image = transforms_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if logits1 is not None:
        return image, label1, label2, logits1, logits2
    else:
        return image, label1

def tensor_to_pil(image, label, logits):
    image = denormalise(image)
    image = transforms_f.to_pil_image(image.cpu())
    label = label.float() / 255.
    label = transforms_f.to_pil_image(label.unsqueeze(0).cpu())
    logits = transforms_f.to_pil_image(logits.unsqueeze(0).cpu())
    return image, label, logits

def tensor_to_pil_2(image, label, logits1, logits2):
    image = denormalise(image)
    image = transforms_f.to_pil_image(image.cpu())
    label = label.float() / 255.
    label = transforms_f.to_pil_image(label.unsqueeze(0).cpu())
    logits1 = transforms_f.to_pil_image(logits1.unsqueeze(0).cpu())
    logits2 = transforms_f.to_pil_image(logits2.unsqueeze(0).cpu())
    return image, label, logits1, logits2

def tensor_to_pil_3(image, label1, label2, logits1, logits2):
    image = denormalise(image)
    image = transforms_f.to_pil_image(image.cpu())
    label1 = label1.float() / 255.
    label1 = transforms_f.to_pil_image(label1.unsqueeze(0).cpu())
    label2 = label2.float() / 255.
    label2 = transforms_f.to_pil_image(label2.unsqueeze(0).cpu())
    logits1 = transforms_f.to_pil_image(logits1.unsqueeze(0).cpu())
    logits2 = transforms_f.to_pil_image(logits2.unsqueeze(0).cpu())
    return image, label1, label2, logits1, logits2

def denormalise(x, imagenet=True):
    if imagenet:
        x = transforms_f.normalize(x, mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225])
        x = transforms_f.normalize(x, mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
        return x
    else:
        return (x + 1) / 2

def batch_transform(images, labels, logits=None, crop_size=(512, 512), scale_size=(0.8, 1.0), augmentation=True):
    image_list, label_list, logits_list = [], [], []
    device = images.device
    for k in range(images.shape[0]):
        image_pil, label_pil, logits_pil = tensor_to_pil(images[k], labels[k], logits[k])
        aug_image, aug_label, aug_logits = transform(image_pil, label_pil, logits_pil, crop_size, scale_size, augmentation)
        image_list.append(aug_image.unsqueeze(0))
        label_list.append(aug_label)
        logits_list.append(aug_logits)
    
    image_trans, label_trans, logits_trans = torch.cat(image_list).to(device), torch.cat(label_list).to(device), torch.cat(logits_list).to(device)
    return image_trans, label_trans, logits_trans

def batch_transform_2(images, labels, logits_1=None, logits_2=None, crop_size=(512, 512), scale_size=(0.8, 1.0), augmentation=True):
    image_list, label_list, logits_1_list, logits_2_list = [], [], [], []
    device = images.device
    for k in range(images.shape[0]):
        image_pil, label_pil, logits_pil_1, logits_pil_2 = tensor_to_pil_2(images[k], labels[k], logits_1[k], logits_2[k])
        aug_image, aug_label, aug_logits_1, aug_logits_2 = transform_2(image_pil, label_pil, logits_pil_1, logits_pil_2, crop_size, scale_size, augmentation)
        image_list.append(aug_image.unsqueeze(0))
        label_list.append(aug_label)
        logits_1_list.append(aug_logits_1)
        logits_2_list.append(aug_logits_2)
    
    image_trans, label_trans, logits_1_trans, logits_2_trans = torch.cat(image_list).to(device), torch.cat(label_list).to(device), torch.cat(logits_1_list).to(device), torch.cat(logits_2_list).to(device)
    return image_trans, label_trans, logits_1_trans, logits_2_trans

def batch_transform_3(images, labels1, labels2, logits_1=None, logits_2=None, crop_size=(512, 512), scale_size=(0.8, 1.0), augmentation=True):
    image_list, label1_list, label2_list, logits_1_list, logits_2_list = [], [], [], [], []
    device = images.device
    for k in range(images.shape[0]):
        image_pil, label1_pil, label2_pil, logits_pil_1, logits_pil_2 = tensor_to_pil_3(images[k], labels1[k], labels2[k], logits_1[k], logits_2[k])
        aug_image, aug_label1, aug_label2, aug_logits_1, aug_logits_2 = transform_3(image_pil, label1_pil, label2_pil, logits_pil_1, logits_pil_2, crop_size, scale_size, augmentation)
        image_list.append(aug_image.unsqueeze(0))
        label1_list.append(aug_label1)
        label2_list.append(aug_label2)
        logits_1_list.append(aug_logits_1)
        logits_2_list.append(aug_logits_2)
    
    image_trans, label1_trans, label2_trans, logits_1_trans, logits_2_trans = torch.cat(image_list).to(device), torch.cat(label1_list).to(device), torch.cat(label2_list).to(device), torch.cat(logits_1_list).to(device), torch.cat(logits_2_list).to(device)
    return image_trans, label1_trans, label2_trans, logits_1_trans, logits_2_trans

def generate_cut_gather(image: torch.Tensor, label: torch.Tensor, logits: torch.Tensor, mode='cutout'):
    
    batch_size, _, image_h, image_w = image.shape
    image = concat_all_gather(image)
    label = concat_all_gather(label)
    logits = concat_all_gather(logits)
    total_size = image.shape[0]
    device = image.device
    rank = dist.get_rank()

    if mode == 'none':
        return image[rank * batch_size: (rank + 1) * batch_size], label[rank * batch_size: (rank + 1) * batch_size].long(), logits[rank * batch_size: (rank + 1) * batch_size]

    new_image = []
    new_label = []
    new_logits = []
    for i in range(total_size):
        if mode == 'cutout': # label: generated region is masked by -1, image: generated region is masked by 0
            mix_mask: torch.Tensor = generate_cutout_mask([image_h, image_w], ratio=2).to(device)
            label[i][(1 - mix_mask).bool()] = -1

            new_image.append((image[i] * mix_mask).unsqueeze(0))
            new_label.append(label[i].unsqueeze(0))
            new_logits.append((logits[i] * mix_mask).unsqueeze(0))
            continue
        elif mode == 'cutmix':
            mix_mask = generate_cutout_mask([image_h, image_w]).to(device)
        elif mode == 'classmix':
            mix_mask = generate_class_mask(label[i]).to(device)
        else:
            raise ValueError('mode must be in cutout, cutmix, or classmix')

        new_image.append((image[i] * mix_mask + image[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_label.append((label[i] * mix_mask + label[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_logits.append((logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
    new_image, new_label, new_logits = torch.cat(new_image), torch.cat(new_label), torch.cat(new_logits)

    return new_image[rank * batch_size: (rank + 1) * batch_size], new_label[rank * batch_size: (rank + 1) * batch_size].long(), new_logits[rank * batch_size: (rank + 1) * batch_size]

def generate_cut_gather_2(image: torch.Tensor, label: torch.Tensor, logits1: torch.Tensor, logits2: torch.Tensor, mode='cutout'):
    
    batch_size, _, image_h, image_w = image.shape
    image = concat_all_gather(image)
    label = concat_all_gather(label)
    logits1 = concat_all_gather(logits1)
    logits2 = concat_all_gather(logits2)
    total_size = image.shape[0]
    device = image.device
    rank = dist.get_rank()

    if mode == 'none':
        return image[rank * batch_size: (rank + 1) * batch_size], label[rank * batch_size: (rank + 1) * batch_size].long(), logits1[rank * batch_size: (rank + 1) * batch_size], logits2[rank * batch_size: (rank + 1) * batch_size]

    new_image = []
    new_label = []
    new_logits1 = []
    new_logits2 = []
    for i in range(total_size):
        if mode == 'cutout': # label: generated region is masked by -1, image: generated region is masked by 0
            mix_mask: torch.Tensor = generate_cutout_mask([image_h, image_w], ratio=2).to(device)
            label[i][(1 - mix_mask).bool()] = -1

            new_image.append((image[i] * mix_mask).unsqueeze(0))
            new_label.append(label[i].unsqueeze(0))
            new_logits1.append((logits1[i] * mix_mask).unsqueeze(0))
            new_logits2.append((logits2[i] * mix_mask).unsqueeze(0))
            continue
        elif mode == 'cutmix':
            mix_mask = generate_cutout_mask([image_h, image_w]).to(device)
        elif mode == 'classmix':
            mix_mask = generate_class_mask(label[i]).to(device)
        else:
            raise ValueError('mode must be in cutout, cutmix, or classmix')

        new_image.append((image[i] * mix_mask + image[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_label.append((label[i] * mix_mask + label[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_logits1.append((logits1[i] * mix_mask + logits1[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_logits2.append((logits2[i] * mix_mask + logits2[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
    new_image, new_label, new_logits1, new_logits2 = torch.cat(new_image), torch.cat(new_label), torch.cat(new_logits1), torch.cat(new_logits2)

    return new_image[rank * batch_size: (rank + 1) * batch_size], new_label[rank * batch_size: (rank + 1) * batch_size].long(), new_logits1[rank * batch_size: (rank + 1) * batch_size], new_logits2[rank * batch_size: (rank + 1) * batch_size]

def generate_cut_gather_3(image: torch.Tensor, label1: torch.Tensor, label2: torch.Tensor, logits1: torch.Tensor, logits2: torch.Tensor, mode='cutout'):
    
    batch_size, _, image_h, image_w = image.shape
    image = concat_all_gather(image)
    label1 = concat_all_gather(label1)
    label2 = concat_all_gather(label2)
    logits1 = concat_all_gather(logits1)
    logits2 = concat_all_gather(logits2)
    total_size = image.shape[0]
    device = image.device
    rank = dist.get_rank()

    new_image = []
    new_label1 = []
    new_label2 = []
    new_logits1 = []
    new_logits2 = []
    for i in range(total_size):
        if mode == 'cutout': # label: generated region is masked by -1, image: generated region is masked by 0
            mix_mask: torch.Tensor = generate_cutout_mask([image_h, image_w], ratio=2).to(device)
            label1[i][(1 - mix_mask).bool()] = -1

            new_image.append((image[i] * mix_mask).unsqueeze(0))
            new_label1.append(label1[i].unsqueeze(0))
            new_logits1.append((logits1[i] * mix_mask).unsqueeze(0))
            new_logits2.append((logits2[i] * mix_mask).unsqueeze(0))
            continue
        elif mode == 'cutmix':
            mix_mask = generate_cutout_mask([image_h, image_w]).to(device)
        elif mode == 'classmix':
            mix_mask = generate_class_mask(label1[i]).to(device)
        else:
            raise ValueError('mode must be in cutout, cutmix, or classmix')

        new_image.append((image[i] * mix_mask + image[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_label1.append((label1[i] * mix_mask + label1[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_label2.append((label2[i] * mix_mask + label2[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_logits1.append((logits1[i] * mix_mask + logits1[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_logits2.append((logits2[i] * mix_mask + logits2[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
    new_image, new_label1, new_label2, new_logits1, new_logits2 = torch.cat(new_image), torch.cat(new_label1), torch.cat(new_label2), torch.cat(new_logits1), torch.cat(new_logits2)

    return new_image[rank * batch_size: (rank + 1) * batch_size], new_label1[rank * batch_size: (rank + 1) * batch_size].long(), new_label2[rank * batch_size: (rank + 1) * batch_size].long(), new_logits1[rank * batch_size: (rank + 1) * batch_size], new_logits2[rank * batch_size: (rank + 1) * batch_size]

def generate_cut(image: torch.Tensor, label: torch.Tensor, logits: torch.Tensor, mode='cutout'):
    if mode == 'none':
        return image, label.long(), logits
    batch_size, _, image_h, image_w = image.shape
    device = image.device

    new_image = []
    new_label = []
    new_logits = []
    for i in range(batch_size):
        if mode == 'cutout': # label: generated region is masked by -1, image: generated region is masked by 0
            mix_mask: torch.Tensor = generate_cutout_mask([image_h, image_w], ratio=2).to(device)
            label[i][(1 - mix_mask).bool()] = -1

            new_image.append((image[i] * mix_mask).unsqueeze(0))
            new_label.append(label[i].unsqueeze(0))
            new_logits.append((logits[i] * mix_mask).unsqueeze(0))
            continue
        elif mode == 'cutmix':
            mix_mask = generate_cutout_mask([image_h, image_w]).to(device)
        elif mode == 'classmix':
            mix_mask = generate_class_mask(label[i]).to(device)
        else:
            raise ValueError('mode must be in cutout, cutmix, or classmix')

        new_image.append((image[i] * mix_mask + image[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_label.append((label[i] * mix_mask + label[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_logits.append((logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
    new_image, new_label, new_logits = torch.cat(new_image), torch.cat(new_label), torch.cat(new_logits)

    return new_image, new_label.long(), new_logits

def generate_class_mask(pseudo_labels: torch.Tensor):
    # select the half classes and cover up them
    labels = torch.unique(pseudo_labels) # all unique labels
    labels_select: torch.Tensor = labels[torch.randperm(len(labels))][:len(labels) // 2] # Randmoly select half of labels
    mask = (pseudo_labels.unsqueeze(-1) == labels_select).any(dim=-1)
    return mask.float()

def generate_cutout_mask(image_size, ratio=2):
    # Cutout: random generate mask where the region inside is 0, one ouside is 1
    cutout_area = image_size[0] * image_size[1] / ratio
    
    w = np.random.randint(image_size[1] / ratio + 1, image_size[1])
    h = np.round(cutout_area / w)

    x_start = np.random.randint(0, image_size[1] - w + 1)
    y_start = np.random.randint(0, image_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = torch.ones(image_size)
    mask[y_start: y_end, x_start: x_end] = 0

    return mask.float()

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