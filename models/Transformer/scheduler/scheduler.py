import torch

class StepLR(torch.optim.lr_scheduler.StepLR):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        super(StepLR, self).__init__(optimizer=optimizer,
                                     step_size=step_size,
                                     gamma=gamma,
                                     last_epoch=last_epoch)
