import torch
from torch import nn
from .diff_tricks import STE, PWL, PWL_C, get_diff_func


def round_clip(input,min_val,max_val):
    return torch.clip(torch.round(input),min_val,max_val)

def bitloss(bit,target):
    pass

class Quantization(nn.Module):
    __constants__ = ['bitwidth', 'gradient_ratio', 'freeze_bitwidth', 'target_bitwidth', 
                     'max_bitwidth','backward_type']
    __backwards__ = {'ste':STE,'pwl':PWL,'pwl-c':PWL_C}
    def __init__(self,bitwidth=8,
                 gradient_ratio=0.,
                 freeze_bitwidth=True,
                 target_bitwidth=4,
                 max_bitwidth=8,
                 backward_type='ste'):
        super(Quantization, self).__init__()
        self.bitwidth=bitwidth
        self.gradient_ratio=gradient_ratio
        self.freeze_bitwidth=freeze_bitwidth
        self.target_bitwidth=target_bitwidth
        self.max_bitwidth=max_bitwidth
        self.backward_type = backward_type
        self.diff_func=get_diff_func(self.backward_type)
        
        self.float_bitwidth=nn.Parameter(torch.tensor(float(self.bitwidth)))
        self.shadow_bitwidth=torch.round(self.float_bitwidth)
        
    def forward(self,input):
        return input
    
    def clone_quantizer(self):
        kwargs = dict()
        for key in self.__constants__:
#             kwargs[key]=self.__getattribute__(key) if key in self.__dict__ else None
            if key in self.__dict__:
                kwargs[key]=self.__getattribute__(key)

        new_quantizer = type(self)(**kwargs)
        new_quantizer.load_state_dict(self.state_dict(),strict=False)
        return new_quantizer
        