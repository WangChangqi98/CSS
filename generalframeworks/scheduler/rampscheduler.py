import numpy as np
class RampScheduler(object):

    def __init__(self, begin_epoch, max_epoch, max_value, ramp_mult):
        super().__init__()
        self.begin_epoch = int(begin_epoch)
        self.max_epoch = int(max_epoch)
        self.max_value = float(max_value)
        self.mult = float(ramp_mult)
        self.epoch = 0

    def step(self):
        self.epoch += 1

    @property
    def value(self):
        return self.get_lr(self.epoch, self.begin_epoch, self.max_epoch, self.max_value, self.mult)

    @staticmethod
    def get_lr(epoch, begin_epoch, max_epochs, max_val, mult):
        if epoch < begin_epoch:
            return 0.
        elif epoch >= max_epochs:
            return max_val
        return max_val * np.exp(mult * (1. - float(epoch - begin_epoch) / (max_epochs - begin_epoch)) ** 2)

class RampdownScheduler(object):

    def __init__(self, begin_epoch, max_epoch, current_epoch, max_value, min_value, ramp_mult):
        super().__init__()
        self.begin_epoch = int(begin_epoch)
        self.max_epoch = int(max_epoch)
        self.max_value = float(max_value)
        self.mult = float(ramp_mult)
        self.epoch = current_epoch
        self.min_value = min_value

    def step(self):
        self.epoch += 1

    @property
    def value(self):
        current_value =  self.get_lr(self.epoch, self.begin_epoch, self.max_epoch, self.max_value, self.min_value, self.mult)
        if current_value < self.min_value:
            current_value = self.min_value
        return current_value

    @staticmethod
    def get_lr(epoch, begin_epoch, max_epochs, max_val, min_value, mult):
        if epoch < begin_epoch:
            return 0.
        elif epoch >= max_epochs:
            return min_value
        return max_val * np.exp(mult * (float(epoch - begin_epoch) / (max_epochs - begin_epoch)) ** 2)