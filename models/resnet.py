#!/usr/bin/python

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    ''' 3x3 convolution with padding '''
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    ''' 1x1 convolution '''
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int=1,
        downsample: nn.Module=None,
        norm_layer: nn.Module=None,
        groups: int=1,
        base_width: int=64,
        dilation: int=1,
    ) -> None:
        
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
            
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(
        self,
        x: torch.Tensor, 
    ) -> torch.Tensor:
        
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)

        return out
    

class Bottleneck(nn.Module):
    
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int=1,
        downsample: nn.Module=None,
        groups: int=1,
        base_width: int=64,
        dilation: int=1, 
        norm_layer: nn.Module=None,
        expansion: int=4,
    ) -> None:
        
        super().__init__()
        
        self.expansion = expansion
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    
class ResNet(nn.Module):
    
    def __init__(
        self,
        block: nn.Module,
        layers: list,
        in_channels: int=3,
        zero_init_residual: bool=True,
        groups: int=1,
        width_per_group: int=64,
        replace_stride_with_dilation: list=None,
        norm_layer: nn.Module=None,
    ) -> None:
        
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        self._norm_layer = norm_layer
        
        self.dilation = 1
        
        self.groups = groups
        self.base_width = width_per_group
        self.zero_init_residual = zero_init_residual
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
            
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                'Replace_stride_with_dilation should be None '
                f'or a 3 element tuple, got {replace_stride_with_dilation}'
            )
            
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(
            block, 64, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 64, 128, layers[1], stride=2,
            dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            block, 128, 256, layers[2], stride=2,
            dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(
            block, 256, 512, layers[3], stride=2,
            dilate=replace_stride_with_dilation[2])
        
        self.init_weights()
        
    def init_weights(
        self,
    ) -> None:
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    tensor=m.weight,
                    mode='fan_out',
                    nonlinearity='relu',
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                    
    def _make_layer(
        self,
        block: nn.Module,
        inplanes: int,
        planes: int,
        blocks: int,
        stride: int=1,
        dilate: bool=False,
    ) -> nn.Module:
        
        downsample = None
        norm_layer = self._norm_layer
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
            
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                inplanes, planes,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
                norm_layer=norm_layer,
            )
        )
        inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(
                block(
                    inplanes, planes,
                    stride=1,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)
    
    
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
    
def _resnet(
    arch: str,
    block: nn.Module,
    layers: list,
    **kwargs,
) -> ResNet:
    
    model = ResNet(block, layers, **kwargs)
    return model
    
def resnet18(
    **kwargs,
) -> ResNet:
    
    r'''ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    '''
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], **kwargs)