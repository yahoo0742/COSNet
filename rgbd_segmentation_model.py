import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from deeplab.deeplabv3_encoder import Encoder, DepthEncoder_ResNet, DepthEncoder_Convs

class RGBDSegmentationModel(nn.Module):
    '''
    :param approach_for_depth: add, conc1, conc2, parallel
    '''
    def __init__(self, block, num_blocks_of_layers_4_rgb, num_blocks_of_layers_4_depth, num_classes, all_channel=256, all_dim=60*60, approach_for_depth="conc1"):	#473./8=60	
        super(RGBDSegmentationModel, self).__init__()

        self.encoder = Encoder(3, block, num_blocks_of_layers_4_rgb, num_classes) # rgb encoder
        if approach_for_depth == "padd":
            self.depth_encoder = DepthEncoder_Convs(8)
        if approach_for_depth == "conv_add" or approach_for_depth == "conv_conc2":
            self.depth_encoder = DepthEncoder_Convs(256)
        else:
            self.depth_encoder = DepthEncoder_ResNet(1, block, num_blocks_of_layers_4_depth, num_classes)

        if approach_for_depth == "parallel":
            self.linear_e = nn.Linear(all_channel*2, all_channel*2,bias = False)
            self.channel = all_channel
            self.dim = all_dim
            self.gate = nn.Conv2d(all_channel*2, 1, kernel_size  = 1, bias = False)
            self.gate_s = nn.Sigmoid()
            self.conv1 = nn.Conv2d(all_channel*2*2, all_channel, kernel_size=3, padding=1, bias = False)
            self.conv2 = nn.Conv2d(all_channel*2*2, all_channel, kernel_size=3, padding=1, bias = False)
        else:
            self.linear_e = nn.Linear(all_channel, all_channel,bias = False)
            self.channel = all_channel
            self.dim = all_dim
            self.gate = nn.Conv2d(all_channel, 1, kernel_size  = 1, bias = False)
            self.gate_s = nn.Sigmoid()
            if approach_for_depth == "conc1":
                self.conv1 = nn.Conv2d(all_channel*(2+1), all_channel, kernel_size=3, padding=1, bias = False)
            else:
                # for add or coc2
                self.conv1 = nn.Conv2d(all_channel*2, all_channel, kernel_size=3, padding=1, bias = False)
            self.conv2 = nn.Conv2d(all_channel*2, all_channel, kernel_size=3, padding=1, bias = False)

        self.bn1 = nn.BatchNorm2d(all_channel)
        self.bn2 = nn.BatchNorm2d(all_channel)
        self.prelu = nn.ReLU(inplace=True)
        self.main_classifier1 = nn.Conv2d(all_channel, num_classes, kernel_size=1, bias = True)
        self.main_classifier2 = nn.Conv2d(all_channel, num_classes, kernel_size=1, bias = True)
        self.softmax = nn.Sigmoid()

        self.approach_for_depth = approach_for_depth

        if approach_for_depth == "add" or approach_for_depth == "conv_add":
            self.depth_gate = nn.Conv2d(all_channel, 1, kernel_size  = 1, bias = True)
            self.depth_gate_s = nn.Sigmoid()
        elif approach_for_depth == "conc2":
            self.depth_conv = nn.Conv2d(all_channel*2, all_channel, kernel_size  = 1, bias = True)

        if approach_for_depth == "parallel":
            self.forward = self.forward_parallel
        elif approach_for_depth == "add" or approach_for_depth == "conv_add":
            self.forward = self.forward_add
        elif approach_for_depth == "conc1":
            self.forward = self.forward_concatenate1
        elif approach_for_depth == "conc2" or approach_for_depth == "conv_conc2":
            self.forward = self.forward_concatenate2
        elif approach_for_depth == "padd":
            self.forward = self.forward_post_add


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

    def forward_parallel(self, rgbs_a, rgbs_b, depths_a, depths_b):
        
        input_size = rgbs_a.size()[2:] # H, W

        V_a, labels = self.encoder(rgbs_a)
        V_b, labels = self.encoder(rgbs_b) # N, C, H, W

        D_a, depth_labels_a = self.depth_encoder(depths_a)
        D_b, depth_labels_b = self.depth_encoder(depths_b)

        fea_size = V_b.size()[2:]	# H, W
        all_dim = fea_size[0]*fea_size[1] #H*W
        V_a_flat = V_a.view(-1, V_a.size()[1], all_dim) #N,C,H*W
        V_b_flat = V_b.view(-1, V_b.size()[1], all_dim) #N, C, H*W
        
        D_a_flat = D_a.view(-1, D_a.size()[1], all_dim) # N, C2, H*W
        D_b_flat = D_b.view(-1, D_b.size()[1], all_dim) # N, C2, H*W

        V_a_flat = torch.cat((V_a_flat, D_a_flat), 1) # merge the rgb channels and the depth channel
        V_b_flat = torch.cat((V_b_flat, D_b_flat), 1) # merge the rgb channels and the depth channel

        # S = B W A_transform
        encoder_future_1_flat_t = torch.transpose(V_a_flat,1,2).contiguous()  #N, H*W, C*2
        weighted_encoder_feature_1 = self.linear_e(encoder_future_1_flat_t) # weighted_encoder_feature_1 = encoder_future_1_flat_t * W, [N, H*W, C*2]
        S = torch.bmm(weighted_encoder_feature_1, V_b_flat) # S = weighted_encoder_feature_1 prod encoder_feature_2_flat, [N, H*W, H*W]

        S_row = F.softmax(S.clone(), dim = 1) # every slice along dim 1 will sum to 1, S row-wise
        S_column = F.softmax(torch.transpose(S,1,2),dim=1) # S column-wise

        Z_b = torch.bmm(V_a_flat, S_row).contiguous() #Z_b = V_a_flat prod S_row
        Z_a = torch.bmm(V_b_flat, S_column).contiguous() # Z_a = V_b_flat prod S_column
        
        input1_att = Z_a.view(-1, V_a_flat.size()[1], fea_size[0], fea_size[1]) # [N, C*2, H, W]
        input2_att = Z_b.view(-1, V_b_flat.size()[1], fea_size[0], fea_size[1]) # [N, C*2, H, W]
        input1_mask = self.gate(input1_att)
        input2_mask = self.gate(input2_att)
        input1_mask = self.gate_s(input1_mask)
        input2_mask = self.gate_s(input2_mask)
        input1_att = input1_att * input1_mask
        input2_att = input2_att * input2_mask

        RGBD_feat_a = V_a_flat.view(-1, V_a_flat.size()[1], fea_size[0], fea_size[1]) # features extracted from RGBD from the encoder
        RGBD_feat_b = V_b_flat.view(-1, V_b_flat.size()[1], fea_size[0], fea_size[1]) # features extracted from RGBD from the encoder

        input1_att = torch.cat([input1_att, RGBD_feat_a],1) 
        input2_att = torch.cat([input2_att, RGBD_feat_b],1)
        input1_att  = self.conv1(input1_att )
        input2_att  = self.conv2(input2_att ) 
        input1_att  = self.bn1(input1_att )
        input2_att  = self.bn2(input2_att )
        input1_att  = self.prelu(input1_att )
        input2_att  = self.prelu(input2_att )

        # Segmentation
        x1 = self.main_classifier1(input1_att)
        x2 = self.main_classifier2(input2_att)   
        x1 = F.upsample(x1, input_size, mode='bilinear')  #upsample to the size of input image, scale=8
        x2 = F.upsample(x2, input_size, mode='bilinear')  #upsample to the size of input image, scale=8
        #print("after upsample, tensor size:", x.size())
        x1 = self.softmax(x1)
        x2 = self.softmax(x2)

        return x1, x2, labels  #shape: [N, 1, H, W]


    def forward_post_add(self, rgbs_a, rgbs_b, depths_a):
        input_size = rgbs_a.size()[2:] # H, W

        V_a, labels = self.encoder(rgbs_a)
        V_b, labels = self.encoder(rgbs_b) # N, C, H, W

        fea_size = V_b.size()[2:]	# H, W
        all_dim = fea_size[0]*fea_size[1] #H*W
        V_a_flat = V_a.view(-1, V_a.size()[1], all_dim) #N,C,H*W
        V_b_flat = V_b.view(-1, V_b.size()[1], all_dim) #N, C, H*W

        # S = B W A_transform
        encoder_future_1_flat_t = torch.transpose(V_a_flat,1,2).contiguous()  #N, H*W, C
        weighted_encoder_feature_1 = self.linear_e(encoder_future_1_flat_t) # weighted_encoder_feature_1 = encoder_future_1_flat_t * W, [N, H*W, C]
        S = torch.bmm(weighted_encoder_feature_1, V_b_flat) # S = weighted_encoder_feature_1 prod encoder_feature_2_flat, [N, H*W, H*W]

        S_row = F.softmax(S.clone(), dim = 1) # every slice along dim 1 will sum to 1, S row-wise
        S_column = F.softmax(torch.transpose(S,1,2),dim=1) # S column-wise

        Z_b = torch.bmm(V_a_flat, S_row).contiguous() #Z_b = V_a_flat prod S_row
        Z_a = torch.bmm(V_b_flat, S_column).contiguous() # Z_a = V_b_flat prod S_column
        
        input1_att = Z_a.view(-1, V_a_flat.size()[1], fea_size[0], fea_size[1]) # [N, C, H, W]
        input2_att = Z_b.view(-1, V_b_flat.size()[1], fea_size[0], fea_size[1]) # [N, C, H, W]
        input1_mask = self.gate(input1_att)
        input2_mask = self.gate(input2_att)
        input1_mask = self.gate_s(input1_mask)
        input2_mask = self.gate_s(input2_mask)
        input1_att = input1_att * input1_mask
        input2_att = input2_att * input2_mask

        input1_att = torch.cat([input1_att, V_a],1)
        input2_att = torch.cat([input2_att, V_b],1)
        input1_att  = self.conv1(input1_att )
        input2_att  = self.conv2(input2_att )
        input1_att  = self.bn1(input1_att )
        input2_att  = self.bn2(input2_att )
        input1_att  = self.prelu(input1_att )
        input2_att  = self.prelu(input2_att )

        # Segmentation
        x1 = self.main_classifier1(input1_att)
        x2 = self.main_classifier2(input2_att)   
        x1 = F.upsample(x1, input_size, mode='bilinear')  #upsample to the size of input image, scale=8
        x2 = F.upsample(x2, input_size, mode='bilinear')  #upsample to the size of input image, scale=8

        # extract depth features
        D_a = self.depth_encoder(depths_a) # N, C, H, W
        x1 = torch.add(D_a, x1)

        #print("after upsample, tensor size:", x.size())
        x1 = self.softmax(x1)
        x2 = self.softmax(x2)

        return x1, x2, labels  #shape: [N, 1, H, W]



    def forward_add(self, rgbs_a, rgbs_b, depths_a):
        input_size = rgbs_a.size()[2:] # H, W

        V_a, labels = self.encoder(rgbs_a)
        V_b, labels = self.encoder(rgbs_b) # N, C, H, W

        fea_size = V_b.size()[2:]	# H, W
        all_dim = fea_size[0]*fea_size[1] #H*W
        V_a_flat = V_a.view(-1, V_a.size()[1], all_dim) #N,C,H*W
        V_b_flat = V_b.view(-1, V_b.size()[1], all_dim) #N, C, H*W

        # S = B W A_transform
        encoder_future_1_flat_t = torch.transpose(V_a_flat,1,2).contiguous()  #N, H*W, C
        weighted_encoder_feature_1 = self.linear_e(encoder_future_1_flat_t) # weighted_encoder_feature_1 = encoder_future_1_flat_t * W, [N, H*W, C]
        S = torch.bmm(weighted_encoder_feature_1, V_b_flat) # S = weighted_encoder_feature_1 prod encoder_feature_2_flat, [N, H*W, H*W]

        S_row = F.softmax(S.clone(), dim = 1) # every slice along dim 1 will sum to 1, S row-wise
        S_column = F.softmax(torch.transpose(S,1,2),dim=1) # S column-wise

        Z_b = torch.bmm(V_a_flat, S_row).contiguous() #Z_b = V_a_flat prod S_row
        Z_a = torch.bmm(V_b_flat, S_column).contiguous() # Z_a = V_b_flat prod S_column
        
        input1_att = Z_a.view(-1, V_a_flat.size()[1], fea_size[0], fea_size[1]) # [N, C, H, W]
        input2_att = Z_b.view(-1, V_b_flat.size()[1], fea_size[0], fea_size[1]) # [N, C, H, W]
        input1_mask = self.gate(input1_att)
        input2_mask = self.gate(input2_att)
        input1_mask = self.gate_s(input1_mask)
        input2_mask = self.gate_s(input2_mask)
        input1_att = input1_att * input1_mask
        input2_att = input2_att * input2_mask

        input1_att = torch.cat([input1_att, V_a],1) 
        input2_att = torch.cat([input2_att, V_b],1)
        input1_att  = self.conv1(input1_att )
        input2_att  = self.conv2(input2_att ) 
        input1_att  = self.bn1(input1_att )
        input2_att  = self.bn2(input2_att )
        input1_att  = self.prelu(input1_att )
        input2_att  = self.prelu(input2_att )

        # extract depth features
        D_a = self.depth_encoder(depths_a) # N, C, H, W
        depth_mask = self.depth_gate(D_a)
        depth_mask = self.depth_gate_s(depth_mask)
        D_a = D_a * depth_mask
        # add weighted depth features to RGB features
        input1_att = torch.add(input1_att, D_a)
        # encode(d) * weight + rgb_features

        # Segmentation
        x1 = self.main_classifier1(input1_att)
        x2 = self.main_classifier2(input2_att)   
        x1 = F.upsample(x1, input_size, mode='bilinear')  #upsample to the size of input image, scale=8
        x2 = F.upsample(x2, input_size, mode='bilinear')  #upsample to the size of input image, scale=8
        #print("after upsample, tensor size:", x.size())
        x1 = self.softmax(x1)
        x2 = self.softmax(x2)

        return x1, x2, labels  #shape: [N, 1, H, W]


    def forward_concatenate1(self, rgbs_a, rgbs_b, depths_a): # concatenate
        input_size = rgbs_a.size()[2:] # H, W

        V_a, labels = self.encoder(rgbs_a)
        V_b, labels = self.encoder(rgbs_b) # N, C, H, W

        fea_size = V_b.size()[2:]	# H, W
        all_dim = fea_size[0]*fea_size[1] #H*W
        V_a_flat = V_a.view(-1, V_a.size()[1], all_dim) #N,C,H*W
        V_b_flat = V_b.view(-1, V_b.size()[1], all_dim) #N, C, H*W

        # S = B W A_transform
        encoder_future_1_flat_t = torch.transpose(V_a_flat,1,2).contiguous()  #N, H*W, C
        weighted_encoder_feature_1 = self.linear_e(encoder_future_1_flat_t) # weighted_encoder_feature_1 = encoder_future_1_flat_t * W, [N, H*W, C]
        S = torch.bmm(weighted_encoder_feature_1, V_b_flat) # S = weighted_encoder_feature_1 prod encoder_feature_2_flat, [N, H*W, H*W]

        S_row = F.softmax(S.clone(), dim = 1) # every slice along dim 1 will sum to 1, S row-wise
        S_column = F.softmax(torch.transpose(S,1,2),dim=1) # S column-wise

        Z_b = torch.bmm(V_a_flat, S_row).contiguous() #Z_b = V_a_flat prod S_row
        Z_a = torch.bmm(V_b_flat, S_column).contiguous() # Z_a = V_b_flat prod S_column
        
        input1_att = Z_a.view(-1, V_a_flat.size()[1], fea_size[0], fea_size[1]) # [N, C, H, W]
        input2_att = Z_b.view(-1, V_b_flat.size()[1], fea_size[0], fea_size[1]) # [N, C, H, W]
        input1_mask = self.gate(input1_att)
        input2_mask = self.gate(input2_att)
        input1_mask = self.gate_s(input1_mask)
        input2_mask = self.gate_s(input2_mask)
        input1_att = input1_att * input1_mask
        input2_att = input2_att * input2_mask

        # extract depth features
        D_a = self.depth_encoder(depths_a) # N, C, H, W

        input1_att = torch.cat([input1_att, V_a, D_a],1) 
        input2_att = torch.cat([input2_att, V_b],1)
        input1_att  = self.conv1(input1_att )
        input2_att  = self.conv2(input2_att ) 
        input1_att  = self.bn1(input1_att )
        input2_att  = self.bn2(input2_att )
        input1_att  = self.prelu(input1_att )
        input2_att  = self.prelu(input2_att )

 

        # V_rgbd_a = torch.cat([input1_att, D_a],1)
        # input1_att = self.depth_conv(V_rgbd_a)

        # Segmentation
        x1 = self.main_classifier1(input1_att)
        x2 = self.main_classifier2(input2_att)   
        x1 = F.upsample(x1, input_size, mode='bilinear')  #upsample to the size of input image, scale=8
        x2 = F.upsample(x2, input_size, mode='bilinear')  #upsample to the size of input image, scale=8
        #print("after upsample, tensor size:", x.size())
        x1 = self.softmax(x1)
        x2 = self.softmax(x2)

        return x1, x2, labels  #shape: [N, 1, H, W]


    def forward_concatenate2(self, rgbs_a, rgbs_b, depths_a): # concatenate
        input_size = rgbs_a.size()[2:] # H, W

        V_a, labels = self.encoder(rgbs_a)
        V_b, labels = self.encoder(rgbs_b) # N, C, H, W

        fea_size = V_b.size()[2:]	# H, W
        all_dim = fea_size[0]*fea_size[1] #H*W
        V_a_flat = V_a.view(-1, V_a.size()[1], all_dim) #N,C,H*W
        V_b_flat = V_b.view(-1, V_b.size()[1], all_dim) #N, C, H*W

        # S = B W A_transform
        encoder_future_1_flat_t = torch.transpose(V_a_flat,1,2).contiguous()  #N, H*W, C
        weighted_encoder_feature_1 = self.linear_e(encoder_future_1_flat_t) # weighted_encoder_feature_1 = encoder_future_1_flat_t * W, [N, H*W, C]
        S = torch.bmm(weighted_encoder_feature_1, V_b_flat) # S = weighted_encoder_feature_1 prod encoder_feature_2_flat, [N, H*W, H*W]

        S_row = F.softmax(S.clone(), dim = 1) # every slice along dim 1 will sum to 1, S row-wise
        S_column = F.softmax(torch.transpose(S,1,2),dim=1) # S column-wise

        Z_b = torch.bmm(V_a_flat, S_row).contiguous() #Z_b = V_a_flat prod S_row
        Z_a = torch.bmm(V_b_flat, S_column).contiguous() # Z_a = V_b_flat prod S_column
        
        input1_att = Z_a.view(-1, V_a_flat.size()[1], fea_size[0], fea_size[1]) # [N, C, H, W]
        input2_att = Z_b.view(-1, V_b_flat.size()[1], fea_size[0], fea_size[1]) # [N, C, H, W]
        input1_mask = self.gate(input1_att)
        input2_mask = self.gate(input2_att)
        input1_mask = self.gate_s(input1_mask)
        input2_mask = self.gate_s(input2_mask)
        input1_att = input1_att * input1_mask
        input2_att = input2_att * input2_mask

        # extract depth features
        D_a = self.depth_encoder(depths_a) # N, C, H, W

        input1_att = torch.cat([input1_att, V_a],1) 
        input2_att = torch.cat([input2_att, V_b],1)
        input1_att  = self.conv1(input1_att )
        input2_att  = self.conv2(input2_att ) 
        input1_att  = self.bn1(input1_att )
        input2_att  = self.bn2(input2_att )
        input1_att  = self.prelu(input1_att )
        input2_att  = self.prelu(input2_att )

        V_rgbd_a = torch.cat([input1_att, D_a],1)
        input1_att = self.depth_conv(V_rgbd_a)

        # Segmentation
        x1 = self.main_classifier1(input1_att)
        x2 = self.main_classifier2(input2_att)   
        x1 = F.upsample(x1, input_size, mode='bilinear')  #upsample to the size of input image, scale=8
        x2 = F.upsample(x2, input_size, mode='bilinear')  #upsample to the size of input image, scale=8
        #print("after upsample, tensor size:", x.size())
        x1 = self.softmax(x1)
        x2 = self.softmax(x2)

        return x1, x2, labels  #shape: [N, 1, H, W]
