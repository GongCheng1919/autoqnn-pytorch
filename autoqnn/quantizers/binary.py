import torch
from torch import nn
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
def binary(input):
    alpha = torch.mean(torch.abs(input))
    fixed = torch.sign(input)
    return alpha*fixed

class Binary(Quantization):

    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        self.register_buffer('alpha', torch.tensor(1.0))
        
    def forward(self,input):
        with torch.no_grad():
            self.alpha = torch.mean(torch.abs(input))
            fixed=torch.sign(input)
            output = self.alpha*fixed
        # grad
        output = self.diff_func(input,output,self.gradient_ratio)
        return output

class BinaryAct(Binary):

    def __init__(self,
                 momentum=0.9,
                **kwargs):
        super().__init__(**kwargs)
        self.register_buffer('momentum', torch.clip(torch.tensor(momentum),0,1))
        
    def forward(self,input):
        with torch.no_grad():
            if self.training:
                alpha = torch.mean(torch.abs(input))
                fixed=torch.sign(input)
                output = alpha*fixed
                
                self.alpha=EMA(self.alpha,alpha,self.momentum)
            else:
                fixed=torch.sign(input)
                output = self.alpha*fixed
        # grad
        output = self.diff_func(input,output,self.gradient_ratio)
        return output
    
'''examples:

model, trainloader, testloader = ...

QUANTIZE_MODULE_MAPPINGS = {
    nn.Conv2d: autoqnn.modules.Conv2d,
}
QUANTIZE_MODULE_CONFIGS = {
    "weight_quant": autoqnn.quantizers.Binary(),
    "bias_quant": autoqnn.quantizers.Binary(),
    "act_quant": autoqnn.quantizers.BinaryAct(),
}

# get quantized model
q_model = autoqnn.core.convert(model,mapping=QUANTIZE_MODULE_MAPPINGS, 
                               inplace=True, quantize_config_dict=QUANTIZE_MODULE_CONFIGS)
                               
optim = torch.optim.SGD(q_model.parameters(), lr=0.1, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

for param_group in optim.param_groups:       # 将更新的lr 送入优化器 optimizer 中，进行下一次优化
    param_group['lr'] = .1
autoqnn.core.train_on_batch(train_loader=trainloader, 
                            model=q_model, 
                            criterion=criterion, 
                            optimizer=optim, 
                            metric_meds=[autoqnn.core.top1])
autoqnn.core.validate(testloader,q_model,criterion,
                      metric_meds=[autoqnn.core.top1],
                      print_freq=100)
'''