import torch
from torch import nn
from .diff_tricks import STE, PWL, PWL_C, get_diff_func

class Quantization(nn.Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
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
        self.diff_func=get_diff_func(backward_type)
        
    def forward(self,input):
        return input