import math
from torch import nn
from torch import Tensor
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init

from torch.nn.modules.utils import _single, _pair, _triple
from torch._jit_internal import List
from typing import Optional, List, Tuple, Union

from autoqnn.quantizers import Quantization
class Linear(nn.modules.linear.Linear):
    __linear_constants__=['in_features', 'out_features', #'bias', 
                          'device','dtype']
    __quant_constants__ = ['weight_quant','bias_quant','act_quant','pre_act_quant']
    
    def __init__(self, *args, **kwargs):
        super(Linear, self).__init__(*args, **kwargs)
        self.weight_quant=kwargs.get("weight_quant",Quantization())
        self.bias_quant=kwargs.get("bias_quant",Quantization())
        self.act_quant=kwargs.get("act_quant",Quantization())
        self.pre_act_quant=kwargs.get("act_quant",True)
        
    def forward(self, input: Tensor) -> Tensor:
        if self.pre_act_quant:
            input = self.act_quant(input)
            
        weight = self.weight_quant(self.weight)
        bias = self.bias_quant(self.bias) if self.bias is not None else None
        self.act = F.linear(input, weight, bias)
        if self.pre_act_quant:
            return self.act
        else:
            return self.act_quant(self.act)
        
    @classmethod
    def from_module(cls,mod):
        kwargs=dict()
        if mod is not None:
            if not isinstance(mod,nn.modules.linear.Linear):
                raise TypeError("Object %s is not the subclass of Linear"%mod.__class__.__name__)
            for key in Linear.__linear_constants__:
                kwargs[key]=mod.__getattribute__(key) if key in mod.__dict__ else None
            
            if hasattr(mod,"bias") and mod.bias is not None:
                kwargs['bias']=True
            else:
                kwargs['bias']=False
        
        new_module = cls(**kwargs)
        new_module.load_state_dict(mod.state_dict(),strict=False)
        
        return new_module
    
class Bilinear(nn.modules.linear.Bilinear):
    __linear_constants__=['in1_features', 'in2_features', 'out_features', #'bias', 
                          'device','dtype']
    __quant_constants__ = ['weight_quant','bias_quant','act_quant']
    
    def __init__(self, *args, **kwargs):
        super(Bilinear, self).__init__(*args,**kwargs)
        self.weight_quant=kwargs.get("weight_quant",Quantization())
        self.bias_quant=kwargs.get("bias_quant",Quantization())
        self.act_quant=kwargs.get("act_quant",Quantization())
        
    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        if self.pre_act_quant:
            input1 = self.act_quant(input1)
            input2 = self.act_quant(input2)
            
        weight = self.weight_quant(self.weight)
        bias = self.bias_quant(self.bias) if self.bias is not None else None
        self.act = F.bilinear(input1, input2, weight, bias)
        if self.pre_act_quant:
            return self.act
        else:
            return self.act_quant(self.act)

    @classmethod
    def from_module(cls,mod):
        kwargs=dict()
        if mod is not None:
            if not isinstance(mod,nn.modules.linear.Bilinear):
                raise TypeError("Object %s is not the subclass of Bilinear"%(mod.__class__.__name__))
            for key in Bilinear.__linear_constants__:
                kwargs[key]=mod.__getattribute__(key) if key in mod.__dict__ else None
            if hasattr(mod,"bias") and mod.bias is not None:
                kwargs['bias']=True
            else:
                kwargs['bias']=False
                
        new_module = cls(**kwargs)
        new_module.load_state_dict(mod.state_dict(),strict=False)
        
        return new_module
    

    