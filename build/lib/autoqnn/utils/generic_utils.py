import torch

def EMA(shadow_var,new_var,momentum=0.99):
    with torch.no_grad():
        var=momentum*shadow_var+(1-momentum)*new_var
    return var

def zero_grad(var):
    if var.grad is not None:
        var.grad.detach_()
        var.grad.zero_()
        
def to_list(var):
    return var if isinstance(var,list) else [var]