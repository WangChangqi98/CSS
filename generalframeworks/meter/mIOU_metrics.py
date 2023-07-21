import numpy as np
import torch
import numpy as np
import torch.nn.functional as F

class mIOUMetrics:
        def __init__(self, num_classes, ignore_index, device='cpu'):
                self.num_classes = num_classes
                self.ignore_index = ignore_index
                self.total_area_inter = torch.zeros(size=(num_classes,),dtype=torch.float64).to(device)
                self.total_area_union = torch.zeros(size=(num_classes,),dtype=torch.float64).to(device)
                self.device = device

        def update(self, predict, target):
                # 预处理 将ignore label对应的像素点筛除
                # target = target.squeeze(1)
                # print('t', target.shape, 'p', predict.shape)
                target_mask = (target != self.ignore_index)  # [batch, height, width]筛选出所有需要训练的像素点标签
                target = target[target_mask]  # [num_pixels]
           
                # _, predict = torch.max(predict, dim=1)
                # print('unique', torch.unique(predict), 'shape:', predict.shape)
                # predict = predict.permute(0,2,3,1)
                # predict = torch.nn.functional.one_hot(predict, 19)
                # print('unique1', torch.unique(predict), 'shape:', predict.shape)

                batch, num_class, height, width = predict.size() # 
                predict = predict.permute(0, 2, 3, 1)  # [batch, height, width, num_class]

                # 计算pixel accuracy
                predict = predict[target_mask.unsqueeze(-1).repeat(1, 1, 1, num_class)].view(-1, num_class)
                predict = predict.argmax(dim=1)
                num_pixels = target.numel()
                correct = (predict == target).sum()
                pixel_acc = correct / num_pixels

                # 计算所有类别的mIoU
                predict = predict + 1
                target = target + 1
                intersection = predict * (predict == target).long()
                area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
                area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
                area_label = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
                
                self.total_area_inter += area_inter
                area_union = (area_pred + area_label - area_inter)
                self.total_area_union += area_union
                

        def reset(self):
                self.total_area_inter = torch.zeros(size=(self.num_classes,),dtype=torch.float64).to(self.device)
                self.total_area_union = torch.zeros(size=(self.num_classes,),dtype=torch.float64).to(self.device)
        
        def get_mIOU(self):
                iou = self.total_area_inter / self.total_area_union
                #print(iou)
                miou = torch.mean(iou[~iou.isnan()])
                if torch.isnan(miou).any():
                        print('get_mIOU somthing wrong! nan detects!')
                return miou



    
    
    
    
    
    
    
    
    