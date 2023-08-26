import torch
from torch import nn
import numpy as np
from .quantization import Quantization,round_clip
from .binary import binary
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

class ResQ(Quantization):
    ''''''
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        
        self.shadow_bitwidth=torch.clip(self.shadow_bitwidth,1,32)
        self.register_buffer('alphas', torch.ones(32,)) 
        
    def quantize(self,input):
        log_w=torch.log(torch.abs(input)+1e-7)/torch.log(torch.tensor(2.0))
        sign=torch.sign(input)
        log_q=round_clip(log_w,0,self.max_exp)
        log_shift_q=log_q-1.0*(log_q==0)
        fixed=sign*torch.pow(torch.tensor(2.0),log_shift_q)
        return fixed
        
    def forward(self,input):
        with torch.no_grad():
            output = torch.zeros_like(input)
            for i in range(int(self.shadow_bitwidth.data.numpy())):
                output += binary(input - output)
                self.alphas[i] = torch.mean(torch.abs(input - output))
                
        # grad
        output = self.diff_func(input,output,self.gradient_ratio)
        return output

class ResQAct(ResQ):

    def __init__(self,
                 momentum=0.9,
                **kwargs):
        super().__init__(**kwargs)
        self.register_buffer('momentum', torch.clip(torch.tensor(momentum),0,1))
        
    def forward(self,input):
        with torch.no_grad():
            output = torch.zeros_like(input).to(input.device)
            alphas = torch.ones(32,).to(self.alphas.device)
            if self.training:
                for i in range(int(self.shadow_bitwidth.data.numpy())):
                    output += binary(input - output)
                    alphas[i] = torch.mean(torch.abs(input - output))
                self.alphas=EMA(self.alphas,alphas,self.momentum)
            else:
                for i in range(int(self.shadow_bitwidth.data.numpy())):
                    output += self.alphas[i]*torch.sign(input - output)
                
        # grad
        output = self.diff_func(input,output,self.gradient_ratio)
        return output
    
'''
examples:

QUANTIZE_MODULE_MAPPINGS = {
    nn.Conv2d: autoqnn.modules.Conv2d,
}
QUANTIZE_MODULE_CONFIGS = {
    "weight_quant": autoqnn.quantizers.ResQ(bitwidth=4),
    "bias_quant": autoqnn.quantizers.ResQ(bitwidth=4),
    "act_quant": autoqnn.quantizers.ResQAct(bitwidth=8),
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