import torch
import numpy as np
# pip install thop
from thop import profile
import matplotlib.pyplot as plt

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

class ModuleIO:
    def __init__(self, module):
        self.module = module
        self.hook_handle = None
        self.input = None
        self.output = None

    def hook(self, module, input, output):
        self.input = input
        self.output = output

    def register(self):
        self.hook_handle = self.module.register_forward_hook(self.hook)

    def unregister(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

class ModuleIOs:
    def __init__(self, named_modules):
        self.ModuleIO_dict={}
        for n, m in named_modules:
            self.ModuleIO_dict[n]=ModuleIO(m)
            
    def register(self):
        for n, m in self.ModuleIO_dict.items():
            m.register()
    
    def unregister(self):
        for n, m in self.ModuleIO_dict.items():
            m.unregister()
    
    def draw_differences(self, rows=None, bins=50,splited=True,scaled=True,figsize=(10,5)):
        if rows is None:
            rows = len(self.ModuleIO_dict)
            cols = 2 if splited else 1
        else:
            splited=False
            cols = (len(self.ModuleIO_dict)-1)//rows+1
        
        draw_num = 1
        plt.figure(figsize=figsize)
        for n, m in self.ModuleIO_dict.items(): 
            plt.subplot(rows,cols,draw_num)
            draw_num+=1
            d = m.input[0].detach().cpu().numpy().flatten()
            if scaled:
                d = np.clip(d,d.mean()-3*d.std(),d.mean()+3*d.std())
            plt.hist(d, bins=bins)
            if splited:
                plt.subplot(rows,cols,draw_num)
                draw_num+=1
            plt.hist(m.output.detach().cpu().numpy().flatten(), bins=bins)
        
        plt.tight_layout()

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
    Byte2Bit = 8.0
    obj = module
    batch_size = input_shape[0]
    inputs=torch.randn(*input_shape)
    _flops, _params = profile(module, inputs=(inputs, ),verbose=False)
    _flops = _flops/batch_size
    _mems=_params*(w_bit/Byte2Bit)+get_feature_size(module,inputs)/batch_size*(a_bit/Byte2Bit)
    if verbose:
        print("Flops, Params and Mems of %s is [%.2fGFLOPs,%.2fM, %.2fM]"%(name,_flops/10**9,_params/10**6,_mems/2**20))
    return _flops, _params,_mems

devices={# device_name:[FLOPS, Bandwidth, Power]
     "QS-855+":[1.032*10**12,34.1*2**30,10],
     "QS-888+":[1.72*10**12,51.2*2**30,10],
     "1080ti":[10.616*10**12,484*2**30,250],
     "2080ti":[11.75*10**12,616*2**30,250],
     "3090":[29.28*10**12,936.2*2**30,350],
     "A6000":[31.29*10**12,768*2**30,300],
     "Xeon E5-2678 v3":[1.9/2*10**12,68*2**30,120],
     "Apple A14 Bionic":[1.536*10**12,34.1*2**30,10],
     "Kirin 9000":[2.332*10**12,44*2**30,10],
     }

def get_attainable_FLOPS(model,input_shape,device_key,w_bit=32,a_bit=32,model_name="model"):
    '''
    The candidate devices includes: 
    ['QS-855+', 'QS-888+', '1080ti', '2080ti', '3090', 'A6000', 'Xeon E5-2678 v3',
    'Apple A14 Bionic', 'Kirin 9000']
    '''
    # get model computing intensity
    flops,_,mems=get_flops_params_mems(model,input_shape,32,32,model_name)
    intensity = flops/mems
    # get device computing intensity
    flops,bandwidth,_ = devices.get(device_key)
    device_intensity = flops/bandwidth
    # get attainable performance
    if intensity>=device_intensity:
        attainable_flops = flops
        print("%s is Compute-bound model on %s"%(model_name,device_key))
    else:
        # model_flops/model_mems * device_bandwidth
        attainable_flops = intensity*bandwidth
        print("%s is IO-bound model on %s"%(model_name,device_key))
    return attainable_flops

###################### stat sparsity #######################
def count_zero(w):
    return torch.sum(w == 0).item()

def stats_module_sparsity(module,name):
    if hasattr(module,name):
        w = getattr(module,name)
        sparsity = count_zero(w)/w.numel()
    return sparsity

def stats_sparsity(parameters_to_prune,verbose=1):
    '''
    return: sparsity_list, global_sparsity
    '''
    zero_count = 0
    numel = 0
    sparsity_list = []
    for i, (module,name) in enumerate(parameters_to_prune):
        w=getattr(module,name)
        zero_count += count_zero(w)
        numel += w.numel()
        sparsity = count_zero(w)/w.numel()
        sparsity_list.append(sparsity)
        if verbose:
            print("Sparsity in submodule[{:d}].{:s}: {:.2f}%".format(
            i,name,100.*sparsity))
    global_sparsity = zero_count/numel
    if verbose:
        print("Global sparsity: {:.2f}%".format(100. * global_sparsity))
    return tuple(sparsity_list), global_sparsity