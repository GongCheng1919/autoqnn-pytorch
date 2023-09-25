import torch
from torch import nn
import torch.nn.functional as F

def vgg_block(num_convs, in_channels, out_channels):
    net = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
           nn.ReLU(True),
           nn.BatchNorm2d(out_channels)]
    for i in range(num_convs - 1): # 定义后面的许多层
        net.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        net.append(nn.ReLU(True))
        net.append(nn.BatchNorm2d(out_channels))
    net.append(nn.MaxPool2d(2, 2)) # 定义池化层
    return nn.Sequential(*net)
 
# 下面我们定义一个函数对这个 vgg block 进行堆叠
def vgg_stack(num_convs, channels): # vgg_net = vgg_stack((1, 1, 2, 2, 2), ((3, 64), (64, 128), (128, 256), (256, 512), (512, 512)))
    net = []
    for n, c in zip(num_convs, channels):
        in_c = c[0]
        out_c = c[1]
        net.append(vgg_block(n, in_c, out_c))
    return nn.Sequential(*net)
 
#确定vgg的类型，是vgg11 还是vgg16还是vgg19
# vgg_net = vgg_stack((1, 1, 2, 2, 2), ((3, 64), (64, 128), (128, 256), (256, 512), (512, 512)))
#vgg类
# VGG7_64((2,2,2),((3,64),(64,128),(128,256)))
class VGG(nn.Module):
    def __init__(self,num_convs,channels,out_dim=16):
        super().__init__()
        self.feature = vgg_stack(num_convs,channels)
        self.fc = nn.Sequential(
            nn.Linear(out_dim*channels[-1][1], 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 10))
        
    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
    
class VGG_plane(nn.Module):
    def __init__(self,num_convs,channels):
        super().__init__()
        self.feature = vgg_stack(num_convs,channels)
        self.avg_pool = nn.AvgPool2d(4)
        self.linear = nn.Linear(channels[-1][1], 10)
        
    def forward(self, x):
        x = self.feature(x)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x
    
def vgg7_M(M):
    return VGG((2,2,2),((3,M),(M,2*M),(2*M,4*M)))

def vgg7_M_plane(M):
    return VGG_plane((2,2,2),((3,M),(M,2*M),(2*M,4*M)))
    
def vgg7_64():
    return vgg7_M(64)
    
def vgg7_64_plane():
    return vgg7_M_plane(64)