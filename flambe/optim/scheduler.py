import torch
from flambe.compile import Component


class LRScheduler(torch.optim.lr_scheduler._LRScheduler, Component):

    def state_dict(self):
        state_dict = super().state_dict()
        del state_dict['_schema']
        del state_dict['_saved_kwargs']
        del state_dict['_extensions']
        return state_dict


class LambdaLR(torch.optim.lr_scheduler.LambdaLR, Component):

    def state_dict(self):
        state_dict = super().state_dict()
        del state_dict['_schema']
        del state_dict['_saved_kwargs']
        del state_dict['_extensions']
        return state_dict
