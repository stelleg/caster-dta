import math
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


# Code slightly modified from https://stackoverflow.com/a/73747249
# Modifications entirely for readability - functionality should be identical
class CosineAnnealingWarmRestartsDecay(CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult=1,
                    eta_min=0, last_epoch=-1, decay_rate=1):
        super().__init__(optimizer, T_0, T_mult=T_mult,
                            eta_min=eta_min, last_epoch=last_epoch)
        self.decay_rate = decay_rate
        self.initial_lrs = self.base_lrs
    
    def step(self, epoch=None):
        # If epoch is None, then use the default behavior of incrementing T_cur by 1
        # (assume a full epoch has passed)
        if epoch == None:
            if self.T_cur + 1 == self.T_i:
                self.base_lrs = [base_lr * self.decay_rate for base_lr in self.base_lrs]
        
        # If epoch is not None, then update the learning rates based on the epoch
        # note that epoch may be a non-integer value, but we want the number of cycles
        # so we cast to an integer after performing the relevant divisions
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    n_cycles = int(epoch / self.T_0)
                else:
                    # Code is identical to the one in the parent CosineAnnealingWarmRestarts
                    n_cycles = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
            else:
                n_cycles = 0
            
            self.base_lrs = [initial_lrs * (self.decay_rate**n_cycles) for initial_lrs in self.initial_lrs]

        super().step(epoch)
