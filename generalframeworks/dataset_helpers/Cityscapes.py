import torch.utils.data as data
import torch
import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
import random
from PIL import Image, ImageFilter
import numpy as np

class Cityscapes_Dataset_cache(data.Dataset):
    def __init__(self, root, idx_list, crop_size=(512, 512), scale_size=(0.5, 2.0), augmentation=True, train=True,
                apply_partial=None, partial_seed=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.crop_size = crop_size
        self.augmentation = augmentation
        self.scale_size = scale_size
        self.idx_list = idx_list
        self.apply_partial = apply_partial
        self.partial_seed = partial_seed
        

    def __getitem__(self, index):
        if self.train:
            image_root, city_name = image_root_transform(self.idx_list[index], mode='train')
            image = Image.open(self.root + image_root)
            label_root = label_root_transform(self.idx_list[index], city_name, mode='train')
            label = Image.open(self.root + label_root)
        else:
            image_root, city_name = image_root_transform(self.idx_list[index], mode='val')
            image = Image.open(self.root + image_root)
            label_root = label_root_transform(self.idx_list[index], city_name, mode='val')
            label = Image.open(self.root + label_root)
        image, label = transform(image, label, None, self.crop_size, self.scale_size, self.augmentation)
        return image, label.squeeze(0)
    
    def __len__(self):
        return len(self.idx_list)

class Cityscapes_Dataset(data.Dataset):
    def __init__(self, root, idx_list, crop_size=(512, 512), scale_size=(0.5, 2.0), augmentation=True, train=True):
        self.root = os.path.expanduser(root)
        self.train = train
        self.crop_size = crop_size
        self.augmentation = augmentation
        self.scale_size = scale_size
        self.idx_list = idx_list

    def __getitem__(self, index):
        if self.train:
            image_root, city_name = image_root_transform(self.idx_list[index], mode='train')
            image = Image.open(self.root + image_root)
            label_root = label_root_transform(self.idx_list[index], city_name, mode='train')
            label = Image.open(self.root + label_root)
        else:
            image_root, city_name = image_root_transform(self.idx_list[index], mode='val')
            image = Image.open(self.root + image_root)
            label_root = label_root_transform(self.idx_list[index], city_name, mode='val')
            label = Image.open(self.root + label_root)
        image, label = transform(image, label, None, self.crop_size, self.scale_size, self.augmentation)
        return image, label.squeeze(0)
    
    def __len__(self):
        return len(self.idx_list)
            
class City_BuildData():
    def __init__(self, data_path, txt_path, label_num, seed, crop_size=[512,512]):
        self.data_path = data_path
        self.txt_path = txt_path
        self.label_num = label_num
        self.seed = seed
        self.im_size = [512, 1024]
        self.crop_size = crop_size
        self.num_segments = 19
        self.scale_size = (1.0, 1.0)
        self.train_l_idx, self.train_u_idx, self.test_idx= get_cityscapes_idx_via_txt(self.txt_path, self.label_num, self.seed)
        
    def build(self):
        train_l_dataset = Cityscapes_Dataset(self.data_path, self.train_l_idx, self.crop_size, self.scale_size,
                                             augmentation=True, train=True)
        train_u_dataset = Cityscapes_Dataset(self.data_path, self.train_u_idx, self.crop_size, scale_size=(1.0, 1.0),
                                                augmentation=False, train=True)
        test_dataset = Cityscapes_Dataset(self.data_path, self.test_idx, self.crop_size, scale_size=(1.0, 1.0),augmentation=False,
                                          train=False)
        return train_l_dataset, train_u_dataset, test_dataset

def get_cityscapes_idx_via_txt(root, label_num, seed):
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

def tensor_to_pil(image, label, logits):
    image = denormalise(image)
    image = transforms_f.to_pil_image(image.cpu())
    label = label.float() / 255.
    label = transforms_f.to_pil_image(label.unsqueeze(0).cpu())
    logits = transforms_f.to_pil_image(logits.unsqueeze(0).cpu())
    return image, label, logits

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

def cityscapes_class_map(mask):
    # source: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    mask_map = np.zeros_like(mask)
    mask_map[np.isin(mask, [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30])] = 255
    mask_map[np.isin(mask, [7])] = 0
    mask_map[np.isin(mask, [8])] = 1
    mask_map[np.isin(mask, [11])] = 2
    mask_map[np.isin(mask, [12])] = 3
    mask_map[np.isin(mask, [13])] = 4
    mask_map[np.isin(mask, [17])] = 5
    mask_map[np.isin(mask, [19])] = 6
    mask_map[np.isin(mask, [20])] = 7
    mask_map[np.isin(mask, [21])] = 8
    mask_map[np.isin(mask, [22])] = 9
    mask_map[np.isin(mask, [23])] = 10
    mask_map[np.isin(mask, [24])] = 11
    mask_map[np.isin(mask, [25])] = 12
    mask_map[np.isin(mask, [26])] = 13
    mask_map[np.isin(mask, [27])] = 14
    mask_map[np.isin(mask, [28])] = 15
    mask_map[np.isin(mask, [31])] = 16
    mask_map[np.isin(mask, [32])] = 17
    mask_map[np.isin(mask, [33])] = 18
    return mask_map

def label_root_transform(root: str, name: str, mode: str):
    label_root = root.strip()[0: -12] + '_gtFine_trainIds'
    return '/gtFine/{}/{}/{}.png'.format(mode, name, label_root)

def image_root_transform(root: str, mode: str):
    name = root[0: root.find('_')]
    return '/leftImg8bit/{}/{}/{}.png'.format(mode, name, root), name