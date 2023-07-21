import torch
import math
import numpy as np
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ConfMatrix(object):
    def __init__(self, num_classes, fmt, name='miou'):
        self.name = name
        self.fmt = fmt
        self.num_classes = num_classes
        self.mat = None
        self.temp_mat = None
        self.val = 0
        self.avg = 0

    
    def update(self, pred, target):
        n = self.num_classes
        self.temp_mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)
            self.temp_mat = torch.bincount(inds, minlength=n**2).reshape(n, n)

    
    def __str__(self):
        h = self.mat.float()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        self.avg = torch.mean(iu).item()
        
        h_t = self.temp_mat.float()
        iu_a = torch.diag(h_t) / (h_t.sum(1) + h_t.sum(0) - torch.diag(h_t))
        self.val = torch.mean(iu_a).item()
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

