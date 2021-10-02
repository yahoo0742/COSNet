import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from deeplab.deeplabv3_encoder import Encoder
from deeplab.deeplabv3_encoder import DepthEncoder_ResNetASPP
# from collections import OrderedDict

"""
R: ResNet
A: ASPP
A: Add
DepthEncoder = ResNet + ASPP
FeatureRGB = RGBEncoder(RGB)
FeatureDepth = DepthEncoder(Depth)
Features = Add(FeatureRGB, FeatureDepth)
"""
class RGBDSegmentation_RAA(nn.Module):
    '''
    :param approach_for_depth: add, conc1, conc2, parallel
    '''
    def __init__(self, block, num_blocks_of_layers_4_rgb, num_blocks_of_layers_4_depth, num_classes, all_channel=256, all_dim=60*60):	#473./8=60	
        super(RGBDSegmentation_RAA, self).__init__()

        # For RGB
        self.encoder = Encoder(3, block, num_blocks_of_layers_4_rgb, num_classes) # rgb encoder
        self.rgb_similarity_weights = nn.Linear(all_channel, all_channel,bias = False) #linear_e
        self.gate = nn.Conv2d(all_channel, 1, kernel_size  = 1, bias = False)
        self.gate_s = nn.Sigmoid()
        self.reduce_channels_A = nn.Conv2d(all_channel*2, all_channel, kernel_size=3, padding=1, bias = False)
        self.reduce_channels_B = nn.Conv2d(all_channel*2, all_channel, kernel_size=3, padding=1, bias = False)
        self.bn_A = nn.BatchNorm2d(all_channel)
        self.bn_B = nn.BatchNorm2d(all_channel)
        self.prelu = nn.ReLU(inplace=True)

        # For depth
        self.depth_encoder = DepthEncoder_ResNetASPP(256, block, num_blocks_of_layers_4_depth, num_classes)
        self.depth_similarity_weights = nn.Linear(all_channel, all_channel,bias = False)
        self.depth_gate = nn.Conv2d(all_channel, 1, kernel_size  = 1, bias = True)
        self.depth_gate_s = nn.Sigmoid() # nn.Sigmoid()
        self.depth_reduce_channels = nn.Conv2d(all_channel*2, all_channel, kernel_size=3, padding=1, bias = False)
        self.depth_bn = nn.BatchNorm2d(all_channel)
        self.depth_weights = nn.Conv2d(all_channel, all_channel, kernel_size=1, bias = True)
        # self.prelu = nn.ReLU(inplace=True)

        # Decoder
        self.segmentation_classifier_A = nn.Conv2d(all_channel, num_classes, kernel_size=1, bias = True)
        self.segmentation_classifier_B = nn.Conv2d(all_channel, num_classes, kernel_size=1, bias = True)
        self.softmax = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
                #init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #init.xavier_normal(m.weight.data)
                #m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def get_params(self, subset="none" or "all" or "encoder" or "rgb_attention" or "rgb" or "depth" or "decoder"):
        modules_with_params = []
        if subset == "none":
            return modules_with_params

        # encoder: encoder
        # rgb_attention: rgb_similarity_weights + gate + reduce_channels_A + reduce_channels_B + bn_A + bn_B
        # rgb: encoder + rgb_attention
        # depth: depth_encoder + depth_gate
        # decoder: segmentation_classifier_A + segmentation_classifier_B
        # all: rgb + depth + decoder
        if subset == "encoder" or subset == "rgb" or subset == "all":
            modules_with_params.append(self.encoder)

        if subset == "rgb_attention" or subset == "rgb" or subset == "all":
            modules_with_params.append(self.rgb_similarity_weights)
            modules_with_params.append(self.gate)
            modules_with_params.append(self.reduce_channels_A)
            modules_with_params.append(self.reduce_channels_B)
            modules_with_params.append(self.bn_A)
            modules_with_params.append(self.bn_B)

        if subset == "depth" or subset == "all":
            mods = self.depth_encoder.get_params()
            modules_with_params.extend(mods)
            modules_with_params.append(self.depth_gate)
            modules_with_params.append(self.depth_similarity_weights)
            modules_with_params.append(self.depth_reduce_channels)
            modules_with_params.append(self.depth_bn)
            modules_with_params.append(self.depth_weights)

        if subset == "decoder" or subset == "all":
            modules_with_params.append(self.segmentation_classifier_A)
            modules_with_params.append(self.segmentation_classifier_B)

        return modules_with_params


    def load_state(self, state_dict):
        new_params = self.state_dict().copy()
        # state_dict_new = OrderedDict()
        for k in state_dict:
            if k.startswith("module."):
                # the state was trained from multiple GPUS
                new_key = k[7:] # remove the prefix module.
            else:
                # the state was trained from a single GPU
                new_key = k
            
            if new_key.startswith("encoder.layer5."):
                new_key = new_key.replace("encoder.layer5.", "encoder.aspp.")
            elif new_key.startswith("encoder.main_classifier"):
                pass
            elif new_key.startswith("encoder."):
                new_key = new_key.replace("encoder.", "encoder.backbone.")
            elif new_key.startswith("linear_e."):
                new_key = new_key.replace("linear_e.","rgb_similarity_weights.")
            elif new_key.startswith("conv1."):
                new_key = new_key.replace("conv1.","reduce_channels_A.")
            elif new_key.startswith("conv2."):
                new_key = new_key.replace("conv2.","reduce_channels_B.")
            elif new_key.startswith("bn1."):
                new_key = new_key.replace("bn1.","bn_A.")
            elif new_key.startswith("bn2."):
                new_key = new_key.replace("bn2.","bn_B.")
            elif new_key.startswith("main_classifier1."):
                new_key = new_key.replace("main_classifier1.","segmentation_classifier_A.")
            elif new_key.startswith("main_classifier2."):
                new_key = new_key.replace("main_classifier2.","segmentation_classifier_B.")
            new_params[new_key] = state_dict[k]

        self.load_state_dict(new_params) 


    def forward(self, rgbs_a, rgbs_b, depths_a, depths_b):
        input_size = rgbs_a.size()[2:] # H, W

        # RGB
        V_a, labels = self.encoder(rgbs_a)
        V_b, labels = self.encoder(rgbs_b) # N, C, H, W
        rgb_feat_channels = V_a.size()[1] 
        rgb_feat_hw = V_a.size()[2:]

        all_dim = rgb_feat_hw[0]*rgb_feat_hw[1] #H*W
        V_a_flat = V_a.view(-1, rgb_feat_channels, all_dim) #N, C, H*W
        V_b_flat = V_b.view(-1, rgb_feat_channels, all_dim) #N, C, H*W

        # S = B W A_transform
        V_a_flat_t = torch.transpose(V_a_flat,1,2).contiguous()  #N, H*W, C
        V_a_flat_t = self.rgb_similarity_weights(V_a_flat_t) # V_a_flat_t = V_a_flat_t * W, [N, H*W, C]
        S = torch.bmm(V_a_flat_t, V_b_flat) # S = V_a_flat_t prod V_b_flat, [N, H*W, H*W]

        S_row = F.softmax(S.clone(), dim = 1) # every slice along dim 1 will sum to 1, S row-wise
        S_column = F.softmax(torch.transpose(S,1,2),dim=1) # S column-wise

        Z_b = torch.bmm(V_a_flat, S_row).contiguous() #Z_b = V_a_flat prod S_row
        Z_a = torch.bmm(V_b_flat, S_column).contiguous() # Z_a = V_b_flat prod S_column
        
        Z_a = Z_a.view(-1, rgb_feat_channels, rgb_feat_hw[0], rgb_feat_hw[1]) # [N, C, H, W]
        Z_b = Z_b.view(-1, rgb_feat_channels, rgb_feat_hw[0], rgb_feat_hw[1]) # [N, C, H, W]
        input_mask_a = self.gate(Z_a)
        input_mask_b = self.gate(Z_b)
        input_mask_a = self.gate_s(input_mask_a)
        input_mask_b = self.gate_s(input_mask_b)
        Z_a = Z_a * input_mask_a
        Z_b = Z_b * input_mask_b

        Z_a = torch.cat([Z_a, V_a],1) 
        Z_b = torch.cat([Z_b, V_b],1)
        Z_a  = self.reduce_channels_A(Z_a )
        Z_b  = self.reduce_channels_B(Z_b ) 
        Z_a  = self.bn_A(Z_a )
        Z_b  = self.bn_B(Z_b )
        # Z_a  = self.prelu(Z_a )
        # Z_b  = self.prelu(Z_b )

        # Depth
        D_a = self.depth_encoder(depths_a) # N, C, H, W
        D_b = self.depth_encoder(depths_b) # N, C, H, W
        depth_feat_channels = D_a.shape[1]
        depth_feat_hw = D_a.size()[2:]
        all_dim = depth_feat_hw[0] * depth_feat_hw[1] #H*W
        D_a_flat = D_a.view(-1, depth_feat_channels, all_dim) #N, C, H*W
        D_b_flat = D_b.view(-1, depth_feat_channels, all_dim) #N, C, H*W

        # S = B W A_transform
        D_a_flat_t = torch.transpose(D_a_flat,1,2).contiguous()  #N, H*W, C
        D_a_flat_t = self.depth_similarity_weights(D_a_flat_t) # D_a_flat_t = D_a_flat_t * W, [N, H*W, C]
        D_S = torch.bmm(D_a_flat_t, D_b_flat) # D_S = D_a_flat_t prod D_b_flat, [N, H*W, H*W]

        D_S_row = F.softmax(D_S.clone(), dim = 1) # every slice along dim 1 will sum to 1, S row-wise
        D_S_column = F.softmax(torch.transpose(D_S,1,2),dim=1) # S column-wise

        D_Z_b = torch.bmm(D_a_flat, D_S_row).contiguous() #DZ_b = D_a_flat prod D_S_row
        D_Z_a = torch.bmm(D_b_flat, D_S_column).contiguous() # DZ_a = D_b_flat prod D_S_column

        D_Z_a = D_Z_a.view(-1, depth_feat_channels, depth_feat_hw[0], depth_feat_hw[1]) # [N, C, H, W]
        D_Z_b = D_Z_b.view(-1, depth_feat_channels, depth_feat_hw[0], depth_feat_hw[1]) # [N, C, H, W]
        D_input_mask_a = self.depth_gate(D_Z_a)
        D_input_mask_b = self.depth_gate(D_Z_b)
        D_input_mask_a = self.depth_gate_s(D_input_mask_a)
        D_input_mask_b = self.depth_gate_s(D_input_mask_b)
        D_Z_a = D_Z_a * D_input_mask_a
        D_Z_b = D_Z_b * D_input_mask_b

        D_Z_a = torch.cat([D_Z_a, D_a],1) 
        D_Z_b = torch.cat([D_Z_b, D_b],1)
        D_Z_a  = self.depth_reduce_channels(D_Z_a )
        D_Z_b  = self.depth_reduce_channels(D_Z_b ) 
        D_Z_a  = self.depth_bn(D_Z_a )
        D_Z_b  = self.depth_bn(D_Z_b )
        D_Z_a  = self.depth_weights(D_Z_a)
        D_Z_b  = self.depth_weights(D_Z_b)
        # D_Z_a  = self.prelu(D_Z_a )
        # D_Z_b  = self.prelu(D_Z_b )

        Z_a = torch.add(Z_a, D_Z_a)
        Z_b = torch.add(Z_b, D_Z_b)

        Z_a  = self.prelu(Z_a )
        Z_b  = self.prelu(Z_b )

        # Segmentation
        x1 = self.segmentation_classifier_A(Z_a)
        x2 = self.segmentation_classifier_B(Z_b)   
        x1 = F.upsample(x1, input_size, mode='bilinear')  #upsample to the size of input image, scale=8
        x2 = F.upsample(x2, input_size, mode='bilinear')  #upsample to the size of input image, scale=8
        #print("after upsample, tensor size:", x.size())
        x1 = self.softmax(x1)
        x2 = self.softmax(x2)

        return x1, x2, labels  #shape: [N, 1, H, W]
