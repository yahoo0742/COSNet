import torch
import torch.nn as nn
import deeplab.config as config

#http://torch.ch/blog/2016/02/04/resnets.html
#https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
#https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels, affine=config.k_learnable_affine_parameters)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=config.k_learnable_affine_parameters)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, in_channels, shrank_channels, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        # TODO: assert shrank_channels * self.expansion = in_channels
        self.conv1 = nn.Conv2d(in_channels, shrank_channels, kernel_size=1, stride=stride, bias=False)  # change configurable stride
        self.bn1 = nn.BatchNorm2d(shrank_channels, affine=config.k_learnable_affine_parameters)

        padding = dilation
        self.conv2 = nn.Conv2d(shrank_channels, shrank_channels, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(shrank_channels, affine=config.k_learnable_affine_parameters)

        self.conv3 = nn.Conv2d(shrank_channels, shrank_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(shrank_channels * self.expansion, affine=config.k_learnable_affine_parameters)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # TODO https://arxiv.org/pdf/1603.05027v3.pdf Identity Mappings in Deep Residual Networks puts Relu and BN before the Conv 
        # https://github.com/BIGBALLON/cifar-10-cnn
        identity = x

        out = self.conv1(x) # 1x1
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) # 3x3
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out) # 1x1
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


"""Partial ResNet"""
class ResNet(nn.Module):
    def __init__(self, input_channels, res_block, num_blocks_of_layers, num_classes):
        self.inner_channels = 64
        self.input_channels = input_channels
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(self.input_channels, self.inner_channels, kernel_size=7, stride=2, padding=3, bias=False) # conv7x7, 64
        self.bn1 = nn.BatchNorm2d(self.inner_channels, affine=config.k_learnable_affine_parameters) # BN,64
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change ceil_mode=True

        self.layer1 = self._make_layer(res_block, out_channels=64, num_blocks=num_blocks_of_layers[0])
        self.layer2 = self._make_layer(res_block, out_channels=128, num_blocks=num_blocks_of_layers[1], stride=2)
        self.layer3 = self._make_layer(res_block, out_channels=256, num_blocks=num_blocks_of_layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(res_block, out_channels=512, num_blocks=num_blocks_of_layers[3], stride=1, dilation=4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # TODO: https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html

    def _make_layer(self, res_block, out_channels, num_blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inner_channels != out_channels * res_block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inner_channels, out_channels * res_block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * res_block.expansion, affine=config.k_learnable_affine_parameters))
        for i in downsample._modules['1'].parameters(): # BN layer
            i.requires_grad = False

        layers = []
        layers.append(res_block(self.inner_channels, out_channels, stride, dilation=dilation, downsample=downsample))
        self.inner_channels = out_channels * res_block.expansion

        for i in range(1, num_blocks):
            layers.append(res_block(self.inner_channels, out_channels, dilation=dilation))

        return nn.Sequential(*layers)


    def get_params(self):
        modules_with_params = []
        modules_with_params.append(self.conv1)
        modules_with_params.append(self.bn1)
        modules_with_params.append(self.layer1)
        modules_with_params.append(self.layer2)
        modules_with_params.append(self.layer3)
        modules_with_params.append(self.layer4)
        return modules_with_params


    def forward(self, x):
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.maxpool(z)

        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z) # (N, 512*4, H, W)

        # normal ResNet includes layers below
        # z = self.avgpool(z)
        # z = torch.flatten(z, 1)
        # z = self.fc(z)

        return z