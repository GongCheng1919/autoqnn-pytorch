import torch
from torch import nn
import numpy as np
from .quantization import Quantization,round_clip
from .diff_tricks import STE, PWL, PWL_C, get_diff_func
from ..utils.generic_utils import EMA

DEBUG = True
# class Quantization(nn.Module):
#     __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
#                      'padding_mode', 'output_padding', 'in_channels',
#                      'out_channels', 'kernel_size']
#     __backwards__ = {'ste':STE,'pwl':PWL,'pwl-c':PWL_C}
#     def __init__(self,bitwidth=8,
#                  gradient_ratio=0.,
#                  freeze_bitwidth=True,
#                  target_bitwidth=4,
#                  max_bitwidth=8,
#                  backward_type='ste'):
#         super(Quantization, self).__init__()
#         self.bitwidth=bitwidth
#         self.gradient_ratio=gradient_ratio
#         self.freeze_bitwidth=freeze_bitwidth
#         self.target_bitwidth=target_bitwidth
#         self.max_bitwidth=max_bitwidth
#         self.diff_func=get_diff_func(backward_type)
        
#     def forward(self,input):
#         return input

class uL2Q(Quantization):
    ''''''
    def __init__(self,
                 lambdas=None,
                **kwargs):
        super().__init__(**kwargs)
        max_quant_val = 2**(self.bitwidth-1)-1
        min_quant_val = -2**(self.bitwidth-1)
        if lambdas is None:
            lambdas = np.array([1.5950161625215948, 0.99529008541347519, 
                              0.5872211503939504, 0.3358904980253396,
                              0.18944224088629152, 0.10570877041455067, 
                              0.057082736023857356, 0.030482181036739827])
        _lamb = lambdas[self.bitwidth-1] if self.bitwidth<=8 else 6/(2**(self.bitwidth)-1)
        
        self.register_buffer('lambdas', torch.tensor(lambdas)) 
        self.register_buffer('_lamb', torch.tensor(_lamb)) 
        self.register_buffer('beta', torch.tensor(0.0)) 
        self.register_buffer('alpha', torch.tensor(1.0)) 
        self.register_buffer('max_quant_val', torch.tensor(max_quant_val))
        self.register_buffer('min_quant_val', torch.tensor(min_quant_val))
        
    def forward(self,input):
        with torch.no_grad():
            self.beta=torch.mean(input)
            self.alpha=self._lamb*torch.std(input)
#             fixed=round_clip(input,(self.min_quant_val+0.5)*(self.alpha+1e-7)+self.beta,
#                                    (self.max_quant_val+0.5)*(self.alpha+1e-7)+self.beta)
            fixed=round_clip((input-self.beta)/(self.alpha+1e-7)-0.5,self.min_quant_val,self.max_quant_val)
            output=fixed*self.alpha+self.beta
        # grad
        output = self.diff_func(input,output,self.gradient_ratio)
        return output

class uL2QAct(uL2Q):

    def __init__(self,
                 momentum=0.9,
                **kwargs):
        super().__init__(**kwargs)
        self.register_buffer('momentum', torch.clip(torch.tensor(momentum),0,1))
        
    def forward(self,input):
        with torch.no_grad():
            if self.training:
                beta=torch.mean(input)
                alpha=self._lamb*torch.std(input)
                fixed=round_clip((input-beta)/(alpha+1e-7)-0.5,self.min_quant_val,self.max_quant_val)
                output=fixed*alpha+beta
                self.beta=EMA(self.beta,beta,self.momentum)
                self.alpha=EMA(self.alpha,alpha,self.momentum)
            else:
                fixed=round_clip((input-self.beta)/(self.alpha+1e-7)-0.5,self.min_quant_val,self.max_quant_val)
                output=fixed*self.alpha+self.beta
        # grad
        output = self.diff_func(input,output,self.gradient_ratio)
        return output
    
'''
examples:

QUANTIZE_MODULE_MAPPINGS = {
    nn.Conv2d: autoqnn.modules.Conv2d,
}
QUANTIZE_MODULE_CONFIGS = {
    "weight_quant": autoqnn.quantizers.uL2Q(bitwidth=4),
    "bias_quant": autoqnn.quantizers.uL2Q(bitwidth=4),
    "act_quant": autoqnn.quantizers.uL2QAct(bitwidth=8),
}

q_model = autoqnn.core.convert(model,mapping=QUANTIZE_MODULE_MAPPINGS, 
                               inplace=True, quantize_config_dict=QUANTIZE_MODULE_CONFIGS)
                               
super_act_modules = {}
super_wei_modules = {}
for n, m in q_model.named_modules():
    if isinstance(m,autoqnn.modules.Conv2d):
        super_wei_modules[n]=m.weight_quant
        super_act_modules[n]=m.act_quant
act_mio = ModuleIOs(super_act_modules.items())
wei_mio = ModuleIOs(super_wei_modules.items())
act_mio.register()
wei_mio.register()

autoqnn.core.validate_on_batch(testloader,q_model.cuda(),criterion,
                      metric_meds=[autoqnn.core.top1])
                      
act_mio.unregister()
act_mio.draw_differences(rows=2)
plt.show()

wei_mio.unregister()
wei_mio.draw_differences(rows=2)
plt.show()
'''