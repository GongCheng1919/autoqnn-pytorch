import torch
from torch import nn
import math
from .quantization import Quantization,round_clip
from ..utils.generic_utils import EMA

class FixedQuant(Quantization):
    def __init__(self,*args,**kwargs):
        super(FixedQuant, self).__init__(*args,**kwargs)
        self.register_buffer('exponent', torch.tensor(0))
        
    def forward(self,input):
        if input is None:
             return input
        with torch.no_grad():
            self.exponent = torch.floor(torch.log(torch.max(torch.abs(input)))/math.log(2.0))-(self.bitwidth-2)
            fixed = torch.round(input/torch.pow(2.0,self.exponent))*torch.pow(2.0,self.exponent)
        return self.diff_func(input,fixed,self.gradient_ratio)

class FixedQuantAct(FixedQuant):
    def __init__(self,*args,**kwargs):
        super(FixedQuantAct, self).__init__(*args,**kwargs)
        self.register_buffer('tensor_maxval', torch.tensor(0))
        
    def forward(self,input):
        if input is None:
             return input
        with torch.no_grad():
            maxval=torch.max(torch.abs(input))
            self.tensor_maxval=EMA(self.tensor_maxval,maxval)
            if self.training:
                self.exponent = torch.floor(torch.log(maxval+1e-7)/math.log(2.0))-(self.bitwidth-2)
            else:
                self.exponent = torch.floor(torch.log(self.tensor_maxval+1e-7)/math.log(2.0))-(self.bitwidth-2)
            fixed = torch.round(input/torch.pow(2.0,self.exponent))*torch.pow(2.0,self.exponent)
        return self.diff_func(input,fixed,self.gradient_ratio)
