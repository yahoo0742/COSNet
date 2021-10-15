import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from deeplab.deeplabv3_encoder import Encoder
from deeplab.deeplabv3_encoder import DepthEncoder_ResNetASPP
# from collections import OrderedDict
import gc

class SiameseNetwork_Debug(nn.Module):
    '''
    :param approach_for_depth: add, conc1, conc2, parallel
    '''
    def __init__(self, block, num_blocks_of_layers_4_rgb, num_classes, all_channel=256, all_dim=60*60, no_grad_for_counterpart=True):	#473./8=60	
        super(SiameseNetwork_Debug, self).__init__()

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

        # Decoder
        self.segmentation_classifier_A = nn.Conv2d(all_channel, num_classes, kernel_size=1, bias = True)
        self.segmentation_classifier_B = nn.Conv2d(all_channel, num_classes, kernel_size=1, bias = True)
        self.softmax = nn.Sigmoid()

        self.no_grad_for_counterpart = no_grad_for_counterpart

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
        if self.no_grad_for_counterpart:
            with torch.no_grad():
                V_b, labels = self.encoder(rgbs_b) # N, C, H, W
        else:
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

        del V_a_flat_t

        S_row = F.softmax(S.clone(), dim = 1) # every slice along dim 1 will sum to 1, S row-wise
        S_column = F.softmax(torch.transpose(S,1,2),dim=1) # S column-wise

        del S

        Z_b = torch.bmm(V_a_flat, S_row).contiguous() #Z_b = V_a_flat prod S_row [N, C, H*W]
        Z_a = torch.bmm(V_b_flat, S_column).contiguous() # Z_a = V_b_flat prod S_column [N, C, H*W]

        del V_a_flat, V_b_flat
        del S_row, S_column
        
        Z_a = Z_a.view(-1, rgb_feat_channels, rgb_feat_hw[0], rgb_feat_hw[1]) # [N, C, H, W]
        Z_b = Z_b.view(-1, rgb_feat_channels, rgb_feat_hw[0], rgb_feat_hw[1]) # [N, C, H, W]
        input_mask_a = self.gate(Z_a) #[N, 1, H, W]
        with torch.no_grad():
            input_mask_b = self.gate(Z_b) #[N, 1, H, W]
        input_mask_a = self.gate_s(input_mask_a) #[N, 1, H, W]
        with torch.no_grad():
            input_mask_b = self.gate_s(input_mask_b) #[N, 1, H, W]
        Z_a = Z_a * input_mask_a # [N, C, H, W]
        Z_b = Z_b * input_mask_b # [N, C, H, W]

        Z_a = torch.cat([Z_a, V_a],1) # [N, 2C, H, W]
        Z_b = torch.cat([Z_b, V_b],1) # [N, 2C, H, W]
        Z_a  = self.reduce_channels_A(Z_a ) # [N, C, H, W]
        Z_b  = self.reduce_channels_B(Z_b ) # [N, C, H, W]
        Z_a  = self.bn_A(Z_a ) # [N, C, H, W]
        Z_b  = self.bn_B(Z_b ) # [N, C, H, W]

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
