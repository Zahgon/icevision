import warnings

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math


class WarmupCosineScheduler(_LRScheduler):
    def __init__(
            self,
            optimizer: Optimizer,
            warmup_epochs: int,
            max_epochs: int,
            warmup_start_lr: float = 1e-8,
            eta_min: float = 1e-8,
            last_epoch: int = -1,
    ):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Number of epochs for warmup
            max_epochs (int): Total number of epochs
            warmup_start_lr (float): Starting lr for warmup
            eta_min (float): Minimum lr for cosine annealing
            last_epoch (int): The index of last epoch
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        # Get initial lr for each param group
        self.base_lrs = []
        for param_group in optimizer.param_groups:
            self.base_lrs.append(param_group['lr'])

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list:
        """Calculate learning rate during warmup and cosine annealing"""
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        epoch = self.last_epoch

        if epoch < self.warmup_epochs:
            # Linear warmup
            lr_scale = epoch / self.warmup_epochs
            return [self.warmup_start_lr + lr_scale * (base_lr - self.warmup_start_lr)
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            cosine_epoch = epoch - self.warmup_epochs
            cosine_total = self.max_epochs - self.warmup_epochs
            return [self.eta_min + 0.5 * (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * cosine_epoch / cosine_total))
                    for base_lr in self.base_lrs]