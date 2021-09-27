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
