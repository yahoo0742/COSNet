import torch
import torch.nn as nn
import torch.nn.functional as F

import deeplab.config as config
import deeplab.residual_net as rn

# https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py

class ASPP(nn.Module):
    def __init__(self, input_channels, output_channels, depth, dilation_series, padding_series):
        super(ASPP, self).__init__()
        # ASPP Pooling
        self.mean = nn.AdaptiveAvgPool2d((1,1))
        self.conv= nn.Conv2d(input_channels, depth, kernel_size=1,stride=1)
        self.bn_x = nn.BatchNorm2d(depth)
        self.relu = nn.ReLU(inplace=True)

        self.conv2d_0 = nn.Conv2d(input_channels, depth, kernel_size=1, stride=1) # 1x1
        self.bn_0 = nn.BatchNorm2d(depth)

        self.conv2d_1 = nn.Conv2d(input_channels, depth, kernel_size=3, stride=1, padding=padding_series[0], dilation=dilation_series[0]) # 3x3, atrous 1 
        self.bn_1 = nn.BatchNorm2d(depth)

        self.conv2d_2 = nn.Conv2d(input_channels, depth, kernel_size=3, stride=1, padding=padding_series[1], dilation=dilation_series[1]) # 3x3, atrous 2
        self.bn_2 = nn.BatchNorm2d(depth)

        self.conv2d_3 = nn.Conv2d(input_channels, depth, kernel_size=3, stride=1, padding=padding_series[2], dilation=dilation_series[2]) # 3x3, atrous 3
        self.bn_3 = nn.BatchNorm2d(depth)

        self.bottleneck = nn.Conv2d( depth*5, output_channels, kernel_size=3, padding=1 )
        self.bn = nn.BatchNorm2d(output_channels)
        self.prelu = nn.PReLU()
        #for m in self.conv2d_list:
        #    m.weight.data.normal_(0, 0.01)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            
    # def _make_stage_(self, dilation1, padding1): # not used
    #     Conv = nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=padding1, dilation=dilation1, bias=True)#classes
    #     Bn = nn.BatchNorm2d(256)
    #     Relu = nn.ReLU(inplace=True)
    #     return nn.Sequential(Conv, Bn, Relu)

    def forward(self, x):
        # x.shape == (N, 2048, W', H')  #(36, 19) if (286,144)   / (48, 25) if (381, 192)
        #out = self.conv2d_list[0](x)
        #mulBranches = [conv2d_l(x) for conv2d_l in self.conv2d_list]
        size=x.shape[2:] # W', H'

        # ASPP Pooling
        image_features=self.mean(x) # (N,2048,1,1), 2048 == input_channels
        image_features=self.conv(image_features) # (N,512,1,1), 512 == depth
        image_features = self.bn_x(image_features)
        image_features = self.relu(image_features) # (N,512,1,1)
        image_features=F.upsample(image_features, size=size, mode='bilinear', align_corners=True) # (N, 512, W', H')

        out_0 = self.conv2d_0(x) # (N, 512, W', H')
        out_0 = self.bn_0(out_0) # (N, 512, W', H')
        out_0 = self.relu(out_0) # (N, 512, W', H')

        out_1 = self.conv2d_1(x) # (N, 512, W', H')
        out_1 = self.bn_1(out_1) # (N, 512, W', H')
        out_1 = self.relu(out_1) # (N, 512, W', H')

        out_2 = self.conv2d_2(x) # (N, 512, W', H')
        out_2 = self.bn_2(out_2) # (N, 512, W', H')
        out_2 = self.relu(out_2) # (N, 512, W', H')

        out_3 = self.conv2d_3(x) # (N, 512, W', H')
        out_3 = self.bn_3(out_3) # (N, 512, W', H')
        out_3 = self.relu(out_3) # (N, 512, W', H')

        out = torch.cat([image_features, out_0, out_1, out_2, out_3], 1) # (N, 2560, W', H'), 512 + 512 + 521 + 512 + 512
        out = self.bottleneck(out) # (N, 256, W', H')
        out = self.bn(out) # (N, 256, W', H')
        out = self.prelu(out) # (N, 256, W', H')
        #for i in range(len(self.conv2d_list) - 1):
        #    out += self.conv2d_list[i + 1](x)
        
        return out


class Encoder(nn.Module):
    def __init__(self, input_channels, res_block, num_blocks_of_layers, num_classes):
        self.inner_channels = 64
        self.input_channels = input_channels
        super(Encoder, self).__init__()

        self.backbone = rn.ResNet(input_channels, res_block, num_blocks_of_layers, num_classes)

        dilations = [ 6, 12, 18]
        paddings = [6, 12, 18]
        self.aspp = ASPP(input_channels=2048, output_channels=256, depth=512, dilation_series=dilations, padding_series=paddings)

        self.main_classifier = nn.Conv2d(256, num_classes, kernel_size=1)
        self.softmax = nn.Sigmoid()#nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        input_size = x.size()[2:] # H, W

        features = self.backbone(x)
        features = self.aspp(features)

        annotation = self.main_classifier(features)
        #print("before upsample, tensor size:", x.size())
        annotation = F.upsample(annotation, input_size, mode='bilinear')  #upsample to the size of input image, scale=8
        #print("after upsample, tensor size:", x.size())
        annotation = self.softmax(annotation)
        return features, annotation
