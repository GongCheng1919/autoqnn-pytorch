# Installation
```pip install git+git@gitee.com:GooCee/AutoQNN-pytorch.git```
# Usage

```
from autoqnn.core import convert
from autoqnn.quantizers import FixedQuant,FixedQuantAct
q_model = convert(module,
                  quantize_config_dict={
				   "weight_quant":FixedQuant(bitwidth=4),
				   "act_quant":FixedQuantAct(bitwidth=4)})
```

# Demo

```
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import autoqnn
from torch import nn
from autoqnn.utils import view_module
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.bn1 = nn.BatchNorm2d(6) # 1 input image channel, 6 output channels, 5x5 square convolution kernel

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), (2, 2)) # Max pooling over a (2, 2) window
        return x
module = Net()
q_model = autoqnn.core.convert(module,
                               quantize_config_dict={
                                   "weight_quant":autoqnn.quantizers.FixedQuant(bitwidth=4),
                                   "act_quant":autoqnn.quantizers.FixedQuantAct(bitwidth=4)})
nodes,edges,dot=autoqnn.utils.view_module(q_model,(1,1,32,32))
```


