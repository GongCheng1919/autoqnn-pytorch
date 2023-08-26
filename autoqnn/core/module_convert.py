import torch
from torch import nn
from .. import modules, quantizers
import copy
import warnings
# Default map for swapping float module to quantized ones
DEFAULT_QUANTIZE_MODULE_MAPPINGS = {
    nn.Conv1d: modules.Conv1d,
    nn.Conv2d: modules.Conv2d,
    nn.Conv3d: modules.Conv3d,
    nn.ConvTranspose1d: modules.ConvTranspose1d,
    nn.ConvTranspose2d: modules.ConvTranspose2d,
    nn.ConvTranspose3d: modules.ConvTranspose3d,
    nn.Linear: modules.Linear,
    nn.Bilinear: modules.Bilinear
}
DEFAULT_QUANTIZE_MODULE_CONFIGS = {
    "weight_quant": quantizers.Quantization(),
    "bias_quant": quantizers.Quantization(),
    "act_quant": quantizers.Quantization(),
}
# DEFAULT_QUANTIZE_MODULE_MAPPINGS[nn.Conv1d]
def _remove_qconfig(module):
    for child in module.children():
        _remove_qconfig(child)
    if hasattr(module, "weight_quant"):
        del module.bias_quant
    if hasattr(module, "bias_quant"):
        del module.bias_quant
    if hasattr(module, "act_quant"):
        del module.bias_quant
def convert(module, mapping=None, inplace=False, quantize_config_dict=None):
    mapping = mapping or DEFAULT_QUANTIZE_MODULE_MAPPINGS
    quantize_config_dict = quantize_config_dict or \
                            DEFAULT_QUANTIZE_MODULE_CONFIGS
    if not inplace:
        module=copy.deepcopy(module)
    _convert(
        module, mapping, inplace=True,
        quantize_config_dict=quantize_config_dict)
    return module

def _convert(module, mapping=None, inplace=False, quantize_config_dict=None):
    mapping = mapping or DEFAULT_QUANTIZE_MODULE_MAPPINGS
    quantize_config_dict = quantize_config_dict or \
                            DEFAULT_QUANTIZE_MODULE_CONFIGS
    if not inplace:
        module = copy.deepcopy(module)
    reassign = {}
    for name, mod in module.named_children():
        # both fused modules and observed custom modules are
        # swapped as one unit
        if mod.__class__ in mapping:
            reassign[name] = swap_module(mod, mapping, quantize_config_dict)
        else:
            _convert(mod, mapping, True, quantize_config_dict)
    for key, value in reassign.items():
        module._modules[key] = value
    return module


def swap_module(mod, mapping, quantize_config_dict,remove_old_config=True):
    if mod.__class__ in mapping:
        cls=mapping[mod.__class__]
        new_mod=cls.from_module(mod)
        for new_p,p in zip(new_mod.parameters(),mod.parameters()):
            new_p.data.copy_(p.data)
        for key,value in quantize_config_dict.items():
            if hasattr(new_mod,key):
                if remove_old_config:
                    new_mod.__delattr__(key)
#                 new_mod.__setattr__(key,copy.deepcopy(value))
                new_mod.__setattr__(key,value.clone_quantizer())

            else:
                warnings.warn("%s is not a attribute of %s"%(key,cls.__name__))
         # respect device affinity when swapping modules
        devices = get_unique_devices_(mod)
        assert len(devices) <= 1, (
            "swap_module only works with cpu or single-device CUDA modules, "
            "but got devices {}".format(devices)
        )
        device = next(iter(devices)) if len(devices) > 0 else None
        if device:
            new_mod.to(device)
        return new_mod
    else:
        raise ValueError("Type %s not in quantize mapping"%mod.__class__.__name__)
        
def get_unique_devices_(module):
    return {p.device for p in module.parameters()} | \
        {p.device for p in module.buffers()}