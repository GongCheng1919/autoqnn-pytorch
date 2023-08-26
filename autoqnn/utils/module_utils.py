import torch 
from torch import nn
from torch.autograd import Variable
from graphviz import Digraph
from .generic_utils import to_list
'''
Example:
from torch import nn
from torch.autograd import Variable
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.bn1 = nn.BatchNorm2d(6) # 1 input image channel, 6 output channels, 5x5 square convolution kernel

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), (2, 2)) # Max pooling over a (2, 2) window
        return x
module = Net()
nodes,edges,dot=view_module(module,(1,1,32,32))
print(dot)
'''
def view_module(module,input_shapes,save_path=None):
    inp = [Variable(torch.randn(*shape),requires_grad = True) for shape in to_list(input_shapes)]
    outs = module(*inp)
    # get the output nodes
    output_nodes = (outs.grad_fn,) if not isinstance(outs, tuple) else tuple(o.grad_fn for o in outs)
    params=dict(list(module.named_parameters()) + [('input', inp)])
    param_map = {id(v): k for k, v in params.items()}
    seen=set()
    nodes=dict()
    edges=dict()
    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'
    def viz_node(var):
        if torch.is_tensor(var):
            # note: this used to show .saved_tensors in pytorch0.2, but stopped
            # working as it was moved to ATen and Variable-Tensor merged
            var_str=(str(id(var)),size_to_str(var.size()))
        elif hasattr(var, 'variable'):
            u = var.variable
            if id(u) in param_map:
                name = param_map[id(u)] if params is not None else ''
            else:
                name="weights"
#             print("variable")
#             print(name,":",type(var),"-")
#             print(name,":",type(u),"-",u.size())
#             name = param_map[id(u)] if params is not None else ''
            node_name = '%s\n %s' % (name, size_to_str(u.size()))
            var_str=(str(id(var)), node_name)
        elif var in output_nodes:
            var_str=(str(id(var)), str(type(var).__name__))
        else:
            var_str=(str(id(var)), str(type(var).__name__))
        return var_str
    def viz_edge(var):
        if var not in seen:
            res=viz_node(var)
            nodes[res[0]]=res[1]
            seen.add(var)
        if hasattr(var, 'next_functions'):
            for u in var.next_functions:
                if u[0] is not None:
                    edges[str(id(u[0]))]=str(id(var))
                    viz_edge(u[0])
        if hasattr(var, 'saved_tensors'):
            for t in var.saved_tensors:
                edges[str(id(t))]=str(id(var))
                viz_edge(t)
    # multiple outputs
    for i,o in enumerate(to_list(outs)):
        node_name = 'output%d\n %s' % (i, size_to_str(o.size()))
        nodes[str(id(o))]=node_name
        edges[str(id(o.grad_fn))]=str(id(o))
    for o in output_nodes:
        viz_edge(o)
    dot = Digraph(name="Net", format="png")
    for key in nodes.keys():
        dot.node(key, nodes[key], fillcolor='orange')
    for key in edges.keys():
        dot.edge(key,edges[key],fillcolor='orange')
    dot.render(filename="Net.png")
    return nodes,edges,dot


def clone_module(module, memo=None):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)
    **Description**
    Creates a copy of a module, whose parameters/buffers/submodules
    are created using PyTorch's torch.clone().
    This implies that the computational graph is kept, and you can compute
    the derivatives of the new modules' parameters w.r.t the original
    parameters.
    **Arguments**
    * **module** (Module) - Module to be cloned.
    **Return**
    * (Module) - The cloned module.
    **Example**
    ~~~python
    net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate all the way to net.
    ~~~
    """
    # NOTE: This function might break in future versions of PyTorch.

    # TODO: This function might require that module.forward()
    #       was called in order to work properly, if forward() instanciates
    #       new variables.
    # TODO: We can probably get away with a shallowcopy.
    #       However, since shallow copy does not recurse, we need to write a
    #       recursive version of shallow copy.
    # NOTE: This can probably be implemented more cleanly with
    #       clone = recursive_shallow_copy(model)
    #       clone._apply(lambda t: t.clone())

    if memo is None:
        # Maps original data_ptr to the cloned tensor.
        # Useful when a Module uses parameters from another Module; see:
        # https://github.com/learnables/learn2learn/issues/174
        memo = {}

    # First, create a copy of the module.
    # Adapted from:
    # https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Second, re-write all parameters
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned

    # Third, handle the buffers if necessary
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[param_ptr] = cloned

    # Then, recurse for each submodule
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(clone, 'flatten_parameters'):
        clone = clone._apply(lambda x: x)
    return clone
