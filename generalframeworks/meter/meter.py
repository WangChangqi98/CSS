from multiprocessing.sharedctypes import Value
import torch
from generalframeworks.utils import class2one_hot
import numpy as np


class Meter(object):

    def reset(self):
        # Reset the Meter to default settings
        pass

    def add(self, pred_logits, label):
        # Log a new value to the meter
        pass

    def value(self):
        # Get the value of the meter in the current state
        pass

    def summary(self) -> dict:
        raise NotImplementedError

    def detailed_summary(self) -> dict:
        raise NotImplementedError

class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None
    
    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)
    
    
    def get_metrics(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        up = torch.diag(h)
        down = h.sum(1) + h.sum(0) - torch.diag(h)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h) + 1e-6)
        return torch.mean(iu).item(), acc.item()

    def get_valid_metrics(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        up = torch.diag(h)
        down = h.sum(1) + h.sum(0) - torch.diag(h)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h) + 1e-6)
        num_no_zero = (iu == 0).sum()
        return iu.sum() / (len(iu) - num_no_zero).item(), acc.item()
        

class My_ConfMatrix(Meter):
    def __init__(self, num_classes):
        super(ConfMatrix, self).__init__()
        self.num_classes = num_classes
        self.mat = None
        self.reset()
        self.mIOU = []
        self.Acc = []

    def add(self, pred_logits, label):
        pred_logits = pred_logits.argmax(1).flatten()
        label = label.flatten()
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred_logits.device)
        with torch.no_grad():
            k = (label >= 0) & (label < n)
            inds = n * label[k].to(torch.int64) + pred_logits[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def value(self, mode='mean'):
        h = self.mat.float()
        self.acc = torch.diag(h).sum() / h.sum()
        self.iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        if mode == 'mean':
            return torch.mean(self.iu).item(), self.acc.item()
        else:
            raise ValueError("mode must be in (mean)")

    def reset(self):
        self.mIOU = []
        self.Acc = []

    def summary(self) -> dict:
        mIOU_dct: dict = {}
        Acc_dct: dict = {}
        for c in range(self.num_classes):
            if c != 0:
                mIOU_dct['mIOU_{}'.format(c)] = np.array([self.value(i, mode='all')[0] for i in range(len(self.mIOU))])[
                                                :, c].mean()
                Acc_dct['Acc_{}'.format(c)] = np.array([self.value(i, mode='all')[1] for i in range(len(self.mIOU))])[:,
                                              c].mean()
        return mIOU_dct, Acc_dct









