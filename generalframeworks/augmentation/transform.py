
from PIL import Image, ImageFilter

import torch
from typing import Tuple
from torchvision import transforms
import torchvision.transforms.functional as transform_f
import random
import numpy as np

def batch_transform(image: torch.Tensor, label: torch.Tensor, logits: torch.Tensor, crop_size: Tuple['h', 'w'], scale_size, 
                apply_augmentation=False):
    image_list, label_list, logits_list = [], [], []
    device = image.device

    for k in range(image.shape[0]):
        image_pil, label_pil, logits_pil = tensor_to_pil(image[k], label[k], logits[k])
        aug_image, aug_label, aug_logits = transform(image_pil, label_pil, logits_pil,
                                                    crop_size=crop_size,
                                                    scale_size=scale_size,
                                                    augmentation=apply_augmentation)
        image_list.append(aug_image.unsqueeze(0))
        label_list.append(aug_label)
        logits_list.append(aug_logits)

    image_trans, label_trans, logits_trans = torch.cat(image_list).to(device), torch.cat(label_list).to(device), \
                                            torch.cat(logits_list).to(device)
    return image_trans, label_trans, logits_trans

def tensor_to_pil(image: torch.Tensor, label: torch.Tensor, logits: torch.Tensor):
    image = denormalise(image)
    image = transform_f.to_pil_image(image.cpu())

    label = label.float() / 255.
    label = transform_f.to_pil_image(label.unsqueeze(0).cpu())

    logits = transform_f.to_pil_image(logits.unsqueeze(0).cpu())

    
    return image, label, logits

def tensor_to_pil_1(image: torch.Tensor, label: torch.Tensor, uncertainty_u:torch.Tensor, logits: torch.Tensor, logits_all: torch.Tensor):
    image = denormalise(image)
    image = transform_f.to_pil_image(image.cpu())

    label = label.float() / 255.
    label = transform_f.to_pil_image(label.unsqueeze(0).cpu())
    uncertainty_u = uncertainty_u.float() / 255.
    uncertainty_u = transform_f.to_pil_image(uncertainty_u.unsqueeze(0).cpu())
    logits_all_l = []
    for i in range(logits_all.shape[0]):
        logits_all_l.append(transform_f.to_pil_image(logits_all[i].float().unsqueeze(0).cpu(), mode='F'))

    logits = transform_f.to_pil_image(logits.unsqueeze(0).cpu(), 'F')
    
    return image, label, uncertainty_u, logits, logits_all_l


def denormalise(x, imagenet=True):
    if imagenet:
        x = transform_f.normalize(x, mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        x = transform_f.normalize(x, mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
        return x
    else:
        return (x + 1) / 2

def transform(image, label, logits=None, crop_size=(512, 512), scale_size=(0.8, 1.0), label_fill=255, augmentation=False):
    '''
    Only apply on the 3d image (one image not batch)
    '''
    # Random Rescale image
    raw_w, raw_h = image.size 
    scale_ratio = random.uniform(scale_size[0], scale_size[1])

    resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
    image = transform_f.resize(image, resized_size, Image.NEAREST)
    label = transform_f.resize(label, resized_size, Image.NEAREST)
    if logits is not None:
        logits = transform_f.resize(logits, resized_size, Image.NEAREST)

    # Adding padding if rescaled image size is less than crop size
    if crop_size == -1: # Use original image size without rop or padding
        crop_size = (raw_h, raw_w)
    
    if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
        right_pad, bottom_pad = max(crop_size[1] - resized_size[1], 0), max(crop_size[0] - resized_size[0], 0)
        image = transform_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')
        label = transform_f.pad(label, padding=(0, 0, right_pad, bottom_pad), fill=label_fill, padding_mode='constant')
        if logits is not None:
            logits = transform_f.pad(logits, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')

    # Random Cropping
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = transform_f.crop(image, i, j, h, w)
    label = transform_f.crop(label, i, j, h, w)
    if logits is not None:
        logits = transform_f.crop(logits, i, j, h, w)
    
    if augmentation:
        # Random Color jitter
        if torch.rand(1) > 0.2:
            color_transform = transforms.ColorJitter.get_params((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))
            image = color_transform(image)

        # Rnadmom Gaussian filter
        if torch.rand(1) > 0.5:
            sigma = random.uniform(0.15, 1.15)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        # Random horizontal filpping
        if torch.rand(1) > 0.5:
            image = transform_f.hflip(image)
            label = transform_f.hflip(label)
            if logits is not None:
                logits = transform_f.hflip(logits)

        # Transform to Tensor
    image = transform_f.to_tensor(image)
    label = (transform_f.to_tensor(label) * 255).long()
    label[label == 255] = -1 # incalid pixels are re-mapping to index -1
    if logits is not None:
        logits = transform_f.to_tensor(logits)
    
    # Apply (ImageNet) normalization
    #image = transform_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = transform_f.normalize(image, mean=[0.5], std=[0.299])
    if logits is not None:
        return image, label, logits
    else:
        return image, label

def generate_cut(image: torch.Tensor, label: torch.Tensor, logits: torch.Tensor, mode='cutout'):
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

def generate_class_mask(pseudo_labels: torch.Tensor):
    # select the half classes and cover up them
    labels = torch.unique(pseudo_labels) # all unique labels
    labels_select: torch.Tensor = labels[torch.randperm(len(labels))][:len(labels) // 2] # Randmoly select half of labels
    mask = (pseudo_labels.unsqueeze(-1) == labels_select).any(dim=-1)
    return mask.float()

def batch_transform_1(data, label, uncertainty_u, logits, logits_all, crop_size, scale_size, apply_augmentation):
    data_list, label_list, uncertainty_u_list, logits_list, logits_all_list = [], [], [], [], []
    device = data.device

    for k in range(data.shape[0]):
        data_pil, label_pil, uncertainty_u_pil, logits_pil, logits_all_pil = tensor_to_pil_1(data[k], label[k], uncertainty_u[k], logits[k], logits_all[k])##ok
        aug_data, aug_label, aug_uncertainty_u, aug_logits, aug_logits_all = transform_1(data_pil, label_pil, uncertainty_u_pil, logits_pil, logits_all_pil,
                                                    crop_size=crop_size,
                                                    scale_size=scale_size,
                                                    augmentation=apply_augmentation)
        
        
        tmp = aug_label.squeeze(0).cuda().eq(aug_logits_all.cuda().argmax(0))
        all = tmp.cuda().sum() + (aug_label.cuda() == -1).sum()
        data_list.append(aug_data.unsqueeze(0))
        label_list.append(aug_label)
        uncertainty_u_list.append(aug_uncertainty_u)
        logits_list.append(aug_logits)
        logits_all_list.append(aug_logits_all.unsqueeze(0))
        #ok

    data_trans, label_trans, uncertainty_u_trans, logits_trans, logits_all_trans = \
        torch.cat(data_list).to(device), torch.cat(label_list).to(device), torch.cat(uncertainty_u_list).to(device), torch.cat(logits_list).to(device), torch.cat(logits_all_list).to(device)
    return data_trans, label_trans, uncertainty_u_trans, logits_trans, logits_all_trans

def transform_1(image, label, uncertainty_u=None, logits=None, logits_all=None, crop_size=(512, 512), scale_size=(0.8, 1.0), augmentation=True):
    # Random rescale image
    
    raw_w, raw_h = image.size
    scale_ratio = random.uniform(scale_size[0], scale_size[1])

    resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
    image = transform_f.resize(image, resized_size, Image.BILINEAR)
    label = transform_f.resize(label, resized_size, Image.NEAREST)
    if uncertainty_u is not None:
        uncertainty_u = transform_f.resize(uncertainty_u, resized_size, Image.NEAREST)
    if logits is not None:
        logits = transform_f.resize(logits, resized_size, Image.NEAREST)
    logits_all_l = []
    if logits_all is not None:
        for logits_item in logits_all:
            logits_all_l.append(transform_f.resize(logits_item, resized_size, Image.NEAREST))
        logits_all = logits_all_l
    
    # Add padding if rescaled image size is less than crop size
    if crop_size == -1:  # use original im size without crop or padding
        crop_size = (raw_h, raw_w)

    if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
        ##ok       
        right_pad, bottom_pad = max(crop_size[1] - resized_size[1], 0), max(crop_size[0] - resized_size[0], 0)
        image = transform_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')
        label = transform_f.pad(label, padding=(0, 0, right_pad, bottom_pad), fill=255, padding_mode='constant')
        if uncertainty_u is not None:
            uncertainty_u = transform_f.pad(uncertainty_u, padding=(0, 0, right_pad, bottom_pad), fill=255, padding_mode='constant')
        if logits is not None:
            logits = transform_f.pad(logits, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')
        if logits_all is not None:
            logits_all_l_tmp = []
            for logits_item in logits_all:
                logits_all_l_tmp.append(transform_f.pad(logits_item, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant'))
            logits_all = logits_all_l_tmp
        # ok


    # Random Cropping
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = transform_f.crop(image, i, j, h, w)
    label = transform_f.crop(label, i, j, h, w)
    if uncertainty_u is not None:
        uncertainty_u = transform_f.crop(uncertainty_u, i, j, h, w)
    if logits is not None:
        logits = transform_f.crop(logits, i, j, h, w)
    if logits_all is not None:
        logits_all_l_tmp = []
        for logits_item in logits_all:
            logits_all_l_tmp.append(transform_f.crop(logits_item, i, j, h, w))
        logits_all = logits_all_l_tmp

    if augmentation:
        # Random color jitter
        if torch.rand(1) > 0.2:
            color_transform = transforms.ColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))  # For PyTorch 1.9/TorchVision 0.10 users
            # color_transform = transforms.ColorJitter.get_params((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))
            image = color_transform(image)

        # Random Gaussian filter
        if torch.rand(1) > 0.5:
            sigma = random.uniform(0.15, 1.15)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        # Random horizontal flipping
        if torch.rand(1) > 0.5:
            image = transform_f.hflip(image)
            label = transform_f.hflip(label)
            if uncertainty_u is not None:
                uncertainty_u = transform_f.hflip(uncertainty_u)
            if logits is not None:
                logits = transform_f.hflip(logits)
            if logits_all is not None:
                logits_all_l_tmp = []
                for logits_item in logits_all:
                    logits_all_l_tmp.append(transform_f.hflip(logits_item))
                logits_all = logits_all_l_tmp

    # Transform to tensor
    image = transform_f.to_tensor(image)
    label = (transform_f.to_tensor(label) * 255).long()
    uncertainty_u = (transform_f.to_tensor(uncertainty_u) * 255).long()
    label[label == 255] = -1  # invalid pixels are re-mapped to index -1
    if logits is not None:
        logits = transform_f.to_tensor(logits)
    if uncertainty_u is not None:
        uncertainty_u[uncertainty_u == 255] = -1
    if logits_all is not None:
        logits_all_l_tmp = []
        for logits_item in logits_all:
            logits_all_l_tmp.append(transform_f.to_tensor(logits_item))
        logits_all = torch.cat(logits_all_l_tmp)

    # Apply (ImageNet) normalisation    
    # image = transform_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if logits is not None and uncertainty_u is not None and logits_all is not None:
        return image, label, uncertainty_u, logits, logits_all
    elif logits is not None and uncertainty_u is None:
        return image, label, logits
    elif logits is None and uncertainty_u is not None:
        return image, label, uncertainty_u
    else:
        return image, label

def generate_cut_1(image: torch.Tensor, label: torch.Tensor, logits: torch.Tensor, uncertainty_u: torch.Tensor=None, logits_all=None, mode='cutout'):
    batch_size, _, image_h, image_w = image.shape
    device = image.device

    new_image = []
    new_label = []
    new_uncertainty_u = []
    new_logits = []
    new_logits_all = []
    for i in range(batch_size):
        if mode == 'cutout': # label: generated region is masked by -1, image: generated region is masked by 0
            mix_mask: torch.Tensor = generate_cutout_mask([image_h, image_w], ratio=2).to(device)
            label[i][(1 - mix_mask).bool()] = -1
            if uncertainty_u is not None:
                uncertainty_u[i][(1 - mix_mask).bool()] = 0

            new_image.append((image[i] * mix_mask).unsqueeze(0))
            new_label.append(label[i].unsqueeze(0))
            if uncertainty_u is not None:
                new_uncertainty_u.append(uncertainty_u[i].unsqueeze(0))
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
        if uncertainty_u is not None:
            new_uncertainty_u.append((uncertainty_u[i] * mix_mask + uncertainty_u[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        if logits_all is not None:
            new_logits_all.append((logits_all[i] * mix_mask + logits_all[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_logits.append((logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))

    new_image, new_label, new_logits = torch.cat(new_image), torch.cat(new_label), torch.cat(new_logits)
    
    if uncertainty_u is not None and logits_all is not None:
        new_uncertainty_u = torch.cat(new_uncertainty_u)
        new_logits_all = torch.cat(new_logits_all)
        
        return new_image, new_label.long(), new_uncertainty_u.long(), new_logits, new_logits_all
    else:
        return new_image, new_label.long(), new_logits


def batch_transform_2(data, label, uncertainty_u, logits, crop_size, scale_size, apply_augmentation):
    data_list, label_list, uncertainty_u_list, logits_list = [], [], [], []
    device = data.device

    for k in range(data.shape[0]):
        data_pil, label_pil, logits_pil = tensor_to_pil(data[k], label[k], logits[k])
        aug_data, aug_label, aug_uncertainty_u, aug_logits = transform_2(data_pil, label_pil, uncertainty_u[k].unsqueeze(0), logits_pil,
                                                    crop_size=crop_size,
                                                    scale_size=scale_size,
                                                    augmentation=apply_augmentation)
        data_list.append(aug_data.unsqueeze(0))
        label_list.append(aug_label)
        # uncertainty_u_list.append(aug_uncertainty_u.unsqueeze(0))
        uncertainty_u_list.append(aug_uncertainty_u)
        logits_list.append(aug_logits)

    data_trans, label_trans, uncertainty_u_trans, logits_trans = \
        torch.cat(data_list).to(device), torch.cat(label_list).to(device), torch.cat(uncertainty_u_list).to(device), torch.cat(logits_list).to(device)
    return data_trans, label_trans, uncertainty_u_trans, logits_trans

def transform_2(image, label, uncertainty_u=None, logits=None, crop_size=(512, 512), scale_size=(0.8, 1.0), augmentation=True):
    # Random rescale image
    raw_w, raw_h = image.size
    scale_ratio = random.uniform(scale_size[0], scale_size[1])

    resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
    image = transform_f.resize(image, resized_size, Image.BILINEAR)
    label = transform_f.resize(label, resized_size, Image.NEAREST)
    if uncertainty_u is not None:
        uncertainty_u = transform_f.resize(uncertainty_u, resized_size, Image.NEAREST)
    if logits is not None:
        logits = transform_f.resize(logits, resized_size, Image.NEAREST)

    # Add padding if rescaled image size is less than crop size
    if crop_size == -1:  # use original im size without crop or padding
        crop_size = (raw_h, raw_w)

    if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
        right_pad, bottom_pad = max(crop_size[1] - resized_size[1], 0), max(crop_size[0] - resized_size[0], 0)
        image = transform_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')
        label = transform_f.pad(label, padding=(0, 0, right_pad, bottom_pad), fill=255, padding_mode='constant')
        if uncertainty_u is not None:
            uncertainty_u = transform_f.pad(uncertainty_u, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')
        if logits is not None:
            logits = transform_f.pad(logits, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')

    # Random Cropping
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = transform_f.crop(image, i, j, h, w)
    label = transform_f.crop(label, i, j, h, w)
    if uncertainty_u is not None:
        uncertainty_u = transform_f.crop(uncertainty_u, i, j, h, w)
    if logits is not None:
        logits = transform_f.crop(logits, i, j, h, w)

    if augmentation:
        # Random color jitter
        if torch.rand(1) > 0.2:
            color_transform = transforms.ColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))  # For PyTorch 1.9/TorchVision 0.10 users
            # color_transform = transforms.ColorJitter.get_params((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))
            image = color_transform(image)

        # Random Gaussian filter
        if torch.rand(1) > 0.5:
            sigma = random.uniform(0.15, 1.15)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        # Random horizontal flipping
        if torch.rand(1) > 0.5:
            image = transform_f.hflip(image)
            label = transform_f.hflip(label)
            if uncertainty_u is not None:
                uncertainty_u = transform_f.hflip(uncertainty_u)
            if logits is not None:
                logits = transform_f.hflip(logits)

    # Transform to tensor
    image = transform_f.to_tensor(image)
    label = (transform_f.to_tensor(label) * 255).long()
    label[label == 255] = -1  # invalid pixels are re-mapped to index -1
    if logits is not None:
        logits = transform_f.to_tensor(logits)

    # Apply (ImageNet) normalisation
    image = transform_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if logits is not None and uncertainty_u is not None:
        return image, label, uncertainty_u, logits
    elif logits is not None and uncertainty_u is None:
        return image, label, logits
    elif logits is None and uncertainty_u is not None:
        return image, label, uncertainty_u
    else:
        return image, label

def generate_cut_2(image: torch.Tensor, label: torch.Tensor, logits: torch.Tensor, uncertainty_u: torch.Tensor=None, mode='cutout'):
    batch_size, _, image_h, image_w = image.shape
    device = image.device

    new_image = []
    new_label = []
    new_uncertainty_u = []
    new_logits = []
    for i in range(batch_size):
        if mode == 'cutout': # label: generated region is masked by -1, image: generated region is masked by 0
            mix_mask: torch.Tensor = generate_cutout_mask([image_h, image_w], ratio=2).to(device)
            label[i][(1 - mix_mask).bool()] = -1
            if uncertainty_u is not None:
                uncertainty_u[i][(1 - mix_mask).bool()] = 0

            new_image.append((image[i] * mix_mask).unsqueeze(0))
            new_label.append(label[i].unsqueeze(0))
            if uncertainty_u is not None:
                new_uncertainty_u.append(uncertainty_u[i].unsqueeze(0))
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
        if uncertainty_u is not None:
            new_uncertainty_u.append((uncertainty_u[i] * mix_mask + uncertainty_u[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_logits.append((logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))

    new_image, new_label, new_logits = torch.cat(new_image), torch.cat(new_label), torch.cat(new_logits)
    if uncertainty_u is not None:
        new_uncertainty_u = torch.cat(new_uncertainty_u)
        return new_image, new_label.long(), new_uncertainty_u, new_logits
    else:
        return new_image, new_label.long(), new_logits
