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

class ZoomQ(Quantization):
    ''''''
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        self.register_buffer('alpha', torch.tensor(1.0)) 
        self.register_buffer('max_val', torch.tensor(1))
        self.register_buffer('min_val', torch.tensor(0))
        
    def forward(self,input):
        with torch.no_grad():
            self.max_val=torch.max(input)
            self.min_val=torch.min(input)
            val_num = 2.0**self.bitwidth
            self.alpha=(self.max_val-self.min_val)/(val_num-1e-4)+1e-7
            fixed=torch.round((input-self.min_val-self.alpha/2)/self.alpha)
            output=torch.clip(fixed*self.alpha+self.min_val+self.alpha/2,self.min_val,self.max_val)
        # grad
        output = self.diff_func(input,output,self.gradient_ratio)
        return output

class ZoomQAct(ZoomQ):

    def __init__(self,
                 momentum=0.9,
                **kwargs):
        super().__init__(**kwargs)
        self.register_buffer('momentum', torch.clip(torch.tensor(momentum),0,1))
        
    def forward(self,input):
        with torch.no_grad():
            if self.training:
                max_val=torch.max(input)
                min_val=torch.min(input)
                val_num = 2.0**self.bitwidth
                alpha=(max_val-min_val)/(val_num-1e-4)+1e-7
                fixed=torch.round((input-min_val-alpha/2)/alpha)
                output=torch.clip(fixed*alpha+min_val+alpha/2,min_val,max_val)
                
                self.max_val=EMA(self.max_val,max_val,self.momentum)
                self.min_val=EMA(self.min_val,min_val,self.momentum)
                self.alpha=EMA(self.alpha,alpha,self.momentum)
            else:
                fixed=torch.round((input-self.min_val-self.alpha/2)/self.alpha)
                output=torch.clip(fixed*self.alpha+self.min_val+self.alpha/2,self.min_val,self.max_val)
        # grad
        output = self.diff_func(input,output,self.gradient_ratio)
        return output
    
'''
examples:

QUANTIZE_MODULE_MAPPINGS = {
    nn.Conv2d: autoqnn.modules.Conv2d,
}
QUANTIZE_MODULE_CONFIGS = {
    "weight_quant": autoqnn.quantizers.ZoomQ(bitwidth=4),
    "bias_quant": autoqnn.quantizers.ZoomQ(bitwidth=4),
    "act_quant": autoqnn.quantizers.ZoomQAct(bitwidth=8),
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