# coding=utf-8
import math
import torch
from torch import Tensor
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init

from torch.nn.modules.utils import _single, _pair, _triple
from torch._jit_internal import List
from typing import Optional, List, Tuple, Union

from autoqnn.quantizers import Quantization
class _ConvNd(nn.modules.conv._ConvNd):
    __conv_constants__=['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode','transposed', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size','device','dtype']
    __constants__ = ['weight_quant','bias_quant','act_quant']
    
    def __init__(self, **kwargs):
        super(_ConvNd, self).__init__(**kwargs)
        self.weight_quant=kwargs.get("weight_quant",Quantization())
        self.bias_quant=kwargs.get("bias_quant",Quantization())
        self.act_quant=kwargs.get("act_quant",Quantization())
    
    @classmethod
    def from_module(cls,mod):
        kwargs=dict()
        if mod is not None:
            if not isinstance(mod,nn.modules.conv._ConvNd):
                raise TypeError("Object %s is not the subclass of Conv"%mod.__class__.__name__)
            for key in _ConvNd.__conv_constants__:
                kwargs[key]=mod.__getattribute__(key) if key in mod.__dict__ else None
        return cls(**kwargs)

class Conv1d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',**kwargs):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        kwargs.update(dict(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, dilation=dilation,
            transposed=False,output_padding=_single(0), groups=groups, 
            bias=bias, padding_mode=padding_mode))
        super(Conv1d, self).__init__(**kwargs)

    def forward(self, input):
        weight = self.weight_quant(self.weight)
        bias = self.bias_quant(self.bias) if self.bias is not None else None
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            self.act = F.conv1d(F.pad(input, expanded_padding, mode='circular'),
                            weight, bias, self.stride,
                            _single(0), self.dilation, self.groups)
        self.act = F.conv1d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)
        return self.act_quant(self.act)

class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',**kwargs):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        kwargs.update(dict(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, dilation=dilation,
            transposed=False,output_padding=_pair(0), groups=groups, 
            bias=bias, padding_mode=padding_mode))
        super(Conv2d, self).__init__(**kwargs)

    def conv2d_forward(self, input, weight):
        weight = self.weight_quant(weight)
        bias = self.bias_quant(self.bias) if self.bias is not None else None
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            self.act = F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        self.act = F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)
        return self.act_quant(self.act)

    def forward(self, input):
        return self.conv2d_forward(input, self.weight)

class Conv3d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',**kwargs):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        kwargs.update(dict(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, dilation=dilation,
            transposed=False,output_padding=_triple(0), groups=groups, 
            bias=bias, padding_mode=padding_mode))
        super(Conv3d, self).__init__(**kwargs)

    def forward(self, input):
        weight = self.weight_quant(self.weight)
        bias = self.bias_quant(self.bias) if self.bias is not None else None
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[2] + 1) // 2, self.padding[2] // 2,
                                (self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            self.act = F.conv3d(F.pad(input, expanded_padding, mode='circular'),
                            weight, bias, self.stride, _triple(0),
                            self.dilation, self.groups)
        self.act = F.conv3d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)
        return self.act_quant(self.act)
    
class _ConvTransposeNd(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, **kwargs):
        if padding_mode != 'zeros':
            raise ValueError('Only "zeros" padding mode is supported for {}'.format(self.__class__.__name__))
        super(_ConvTransposeNd, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
            stride=stride,padding=padding, dilation=dilation, transposed=transposed, 
            output_padding=output_padding, groups=groups, bias=bias, 
            padding_mode=padding_mode, **kwargs)

    # dilation being an optional parameter is for backwards
    # compatibility
    def _output_padding(self, input: Tensor, output_size: Optional[List[int]],
                        stride: List[int], padding: List[int], kernel_size: List[int],
                        dilation: Optional[List[int]] = None) -> List[int]:
        if output_size is None:
            ret = _single(self.output_padding)  # converting to list if was not already
        else:
            k = input.dim() - 2
            if len(output_size) == k + 2:
                output_size = output_size[2:]
            if len(output_size) != k:
                raise ValueError(
                    "output_size must have {} or {} elements (got {})"
                    .format(k, k + 2, len(output_size)))

            min_sizes = torch.jit.annotate(List[int], [])
            max_sizes = torch.jit.annotate(List[int], [])
            for d in range(k):
                dim_size = ((input.size(d + 2) - 1) * stride[d] -
                            2 * padding[d] +
                            (dilation[d] if dilation is not None else 1) * (kernel_size[d] - 1) + 1)
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError((
                        "requested an output size of {}, but valid sizes range "
                        "from {} to {} (for an input of {})").format(
                            output_size, min_sizes, max_sizes, input.size()[2:]))

            res = torch.jit.annotate(List[int], [])
            for d in range(k):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret

class ConvTranspose1d(_ConvTransposeNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros',**kwargs):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        output_padding = _single(output_padding)
        kwargs.update(dict(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, dilation=dilation,
            transposed=True,output_padding=output_padding, groups=groups, 
            bias=bias, padding_mode=padding_mode))
        super(ConvTranspose1d, self).__init__(**kwargs)

    def forward(self, input, output_size=None):
        # type: (Tensor, Optional[List[int]]) -> Tensor
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')
        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
        weight = self.weight_quant(self.weight)
        bias = self.bias_quant(self.bias) if self.bias is not None else None
        self.act = F.conv_transpose1d(
            input, weight, bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
        return self.act_quant(self.act)

class ConvTranspose2d(_ConvTransposeNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros',**kwargs):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        kwargs.update(dict(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, dilation=dilation,
            transposed=True,output_padding=output_padding, groups=groups, 
            bias=bias, padding_mode=padding_mode))
        super(ConvTranspose2d, self).__init__(**kwargs)

    def forward(self, input, output_size=None):
        # type: (Tensor, Optional[List[int]]) -> Tensor
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')
        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
        weight = self.weight_quant(self.weight)
        bias = self.bias_quant(self.bias) if self.bias is not None else None
        self.act = F.conv_transpose2d(
            input, weight, bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
        return self.act_quant(self.act)

class ConvTranspose3d(_ConvTransposeNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros',**kwargs):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        output_padding = _triple(output_padding)
        kwargs.update(dict(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, dilation=dilation,
            transposed=True,output_padding=output_padding, groups=groups, 
            bias=bias, padding_mode=padding_mode))
        super(ConvTranspose3d, self).__init__(**kwargs)

    def forward(self, input, output_size=None):
        # type: (Tensor, Optional[List[int]]) -> Tensor
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose3d')
        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
        weight = self.weight_quant(self.weight)
        bias = self.bias_quant(self.bias) if self.bias is not None else None
        self.act = F.conv_transpose3d(
            input, weight, bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
        return self.act_quant(self.act)
