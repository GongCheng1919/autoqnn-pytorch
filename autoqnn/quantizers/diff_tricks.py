import torch
import warnings

def STE(full_var,fixed_tensor,diff_ratio=None):
    return full_var+(fixed_tensor-full_var.detach())

def PWL(full_var,fixed_tensor,diff_ratio):
    with torch.no_grad():
        new_fixed_tensor=fixed_tensor+diff_ratio*(full_var-fixed_tensor)
    return STE(full_var,new_fixed_tensor)

def PWL_C(full_var,fixed_tensor,diff_ratio):
    with torch.no_grad():
        min_val = torch.min(fixed_tensor)
        max_val = torch.max(fixed_tensor)
    full_var=torch.clip(full_var,min_val,max_val)
    return PWL(full_var,fixed_tensor,diff_ratio)

__backwards__ = {'default':STE,'ste':STE,'pwl':PWL,'pwl-c':PWL_C}

def get_diff_func(backwards):
    if backwards not in __backwards__:
        warnings.warn("argument %s not in candidates, using default diff_func. candidates are:"%backwards,list(__backwards__.keys()))
    return __backwards__[backwards] if backwards in __backwards__ else __backwards__['default']