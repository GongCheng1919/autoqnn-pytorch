import torch
from torch import nn
import numpy as np
from .quantization import Quantization,round_clip
from .diff_tricks import STE, PWL, PWL_C, get_diff_func
from ..utils.generic_utils import EMA

class ESB_PerChannel(Quantization):
    ''''''
    def __init__(self,
                 linear_bits=4,
                 mean_shift=True,
                **kwargs):
        super().__init__(**kwargs)
        if self.bitwidth< linear_bits:
            raise ValueError("Error: data_max_bits should greater than linear_bits")
        self.nonlinear_bits=min(4,max(1,self.bitwidth-linear_bits-1)) # 1 bit for sign, no more than 4 bits
        self.linear_bits=self.bitwidth-self.nonlinear_bits-1
        self.mean_shift=mean_shift
        self.extra_args.append('mean_shift')
        self.extra_args.append('linear_bits')

        if self.nonlinear_bits<2: # Linear quantization
            bits=int(self.bitwidth)
            self.lambdas=[1.5958,0.9957,0.5860,0.3352,0.1881,0.1041,0.0569,0.0308]
            self.loc_alpha=self.lambdas[bits-1]
            self.c=2**(bits-1)-0.5
        else:
            '''
            Mixed quantization: Linear and Non-linear quantization
            '''
            bits=int(self.bitwidth)
            linear_bits = int(self.linear_bits)
            self.N=np.floor((2**(self.nonlinear_bits))-1.5)
            self.c=2**(self.N+1)-2**(self.N-self.linear_bits)
            self.alpha_list=[[1.2240], # 2,0
                        [0.5181,1.3015], # 3,0;3,1
                        [0.0381,0.4871,1.4136], # 4,0;4,1;4,2
                        [None,0.0391,0.4828,1.5460], # 5,1;5,2;5,3
                        [None,None,0.0406,0.4997,1.6878], # 6,2;6,3;6,4
                        [None,None,None,0.0409,0.5247,1.8324], # 7,3;7,4;7,5
                        [None,None,None,None,0.0412,0.5527,1.9757], # 8,4;8,5;8,6
                        [None,None,None,None,None,0.0416,0.5816,1.9757]] # 9,5;9,6;9,7
            self.loc_alpha=self.alpha_list[bits-2][linear_bits]
        
        self.register_buffer('alpha_vec', torch.ones(self.channel_size)) 
        self.register_buffer('mean_vec', torch.zeros(self.channel_size)) 
        # self.register_buffer('max_val', torch.ones(self.channel_size))
        # self.register_buffer('min_val', torch.zeros(self.channel_size))
        
    def forward(self,input):
        with torch.no_grad():
            # prepare
            if self.per_channel:
                reduce_dim_tuple = tuple(i for i in range(input.dim()) \
                                        if i != self.channel_axis)
                reduce_shape = tuple(self.channel_size \
                                    if i==self.channel_axis else 1 \
                                    for i in range(input.dim()))
            else:
                reduce_dim_tuple = tuple(range(input.dim()))
                reduce_shape = (1,) * input.dim()
            self.std = torch.std(input,dim=reduce_dim_tuple).view(reduce_shape)
            alpha_vec=self.loc_alpha*self.std+1e-5
            self.alpha_vec.data.copy_(alpha_vec.view(-1))
            if self.mean_shift:
                mean_vec=torch.mean(input,dim=reduce_dim_tuple).view(reduce_shape)
                self.mean_vec.data.copy_(mean_vec.view(-1))
                input=input-mean_vec
            input = input/alpha_vec
            # input=torch.clip(input,-self.c*self.alpha,self.c)
            # quantization
            if self.nonlinear_bits<2: # 1-bit sign and 1-bit exponent, equivalent to the Linear Q with linear_bits+1
                # fixed = torch.round(input/2**(-self.bitwidth-1))*2**(-self.bitwidth-1)
                fixed = torch.round(input-0.5)+0.5
            else:
                ns=torch.floor(torch.log(torch.abs(input)+1e-7)/np.log(2.0))
                ns=torch.clip(ns,0,self.N)
                shifts=torch.pow(2.0,ns-self.linear_bits)
                fixed=torch.round(input/shifts)*shifts
            fixed = torch.clip(fixed,-self.c,self.c)
            output = fixed*alpha_vec
            if self.mean_shift:
                output=output+mean_vec
            # output = fixed
        
        # grad
        output = self.diff_func(input,output,self.gradient_ratio)
        return output