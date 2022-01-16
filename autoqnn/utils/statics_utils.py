import torch
import numpy as np
from thop import profile

# get feature size
class HookTool: 
    def __init__(self):
        self.fea = None 

    def hook_fun(self, module, fea_in, fea_out):
        self.fea = fea_out

def get_feas_by_hook(module):
    fea_hooks = []
    for n, m in module.named_modules():
        cur_hook = HookTool()
        m.register_forward_hook(cur_hook.hook_fun)
        fea_hooks.append(cur_hook)

    return fea_hooks

def get_feature_size(module,inputs):
    fea_hooks = get_feas_by_hook(module)
    out = module(inputs)
    shapes=[]
    for fea_hook in fea_hooks:
        if fea_hook.fea is not None and hasattr(fea_hook.fea,"shape"):
            shapes.append(fea_hook.fea.shape)
    out_size=sum([np.prod(shape) for shape in shapes])
    return out_size

# get model information
def get_flops_params_mems(module,input_shape,w_bit,a_bit,name="module",verbose=1):
    '''
    Example:
        import torchvision
        alexnet = torchvision.models.alexnet()
        get_flops_params_mems(alexnet,input_shape,32,32,"alexnet")
        -------
        >> Flops, Params and Mems of alexnet is [0.71GFLOPs,61.10M, 474.63M]
    '''
    obj = module
    batch_size = input_shape[0]
    inputs=torch.randn(*input_shape)
    _flops, _params = profile(module, inputs=(inputs, ),verbose=False)
    _flops = _flops/batch_size
    _mems=_params*(w_bit/4.0)+get_feature_size(module,inputs)/batch_size*(a_bit/4.0)
    if verbose:
        print("Flops, Params and Mems of %s is [%.2fGFLOPs,%.2fM, %.2fM]"%(name,_flops/10**9,_params/10**6,_mems/2**20))
    return _flops, _params,_mems