import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from deeplab.deeplabv3_encoder import Encoder

class RGBDSegmentationModel(nn.Module):
    def __init__(self, block, num_blocks_of_layers, num_classes, all_channel=256, all_dim=60*60):	#473./8=60	
        super(RGBDSegmentationModel, self).__init__()

        self.encoder = Encoder(3, block, num_blocks_of_layers, num_classes) # rgb encoder
        self.depth_encoder = Encoder(1, block, num_blocks_of_layers, num_classes)

        self.linear_e = nn.Linear(all_channel*2, all_channel*2,bias = False)
        self.channel = all_channel
        self.dim = all_dim
        self.gate = nn.Conv2d(all_channel*2, 1, kernel_size  = 1, bias = False)
        self.gate_s = nn.Sigmoid()
        self.conv1 = nn.Conv2d(all_channel*2*2, all_channel, kernel_size=3, padding=1, bias = False)
        self.conv2 = nn.Conv2d(all_channel*2*2, all_channel, kernel_size=3, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(all_channel)
        self.bn2 = nn.BatchNorm2d(all_channel)
        self.prelu = nn.ReLU(inplace=True)
        self.main_classifier1 = nn.Conv2d(all_channel, num_classes, kernel_size=1, bias = True)
        self.main_classifier2 = nn.Conv2d(all_channel, num_classes, kernel_size=1, bias = True)
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

    
    def forward(self, rgbs_a, rgbs_b, depths_a, depths_b):
        
        input_size = rgbs_a.size()[2:] # H, W
        print("Before encoder size: ",input_size)

        V_a, labels = self.encoder(rgbs_a)
        print("After RGB encoder target")
        V_b, labels = self.encoder(rgbs_b) # N, C, H, W
        print("After RGB encoder match")

        D_a = self.depth_encoder(depths_a)
        print("After depth encoder target")
        D_b = self.depth_encoder(depths_b)
        print("After depth encoder match")


        fea_size = V_b.size()[2:]	# H, W
        all_dim = fea_size[0]*fea_size[1] #H*W
        V_a_flat = V_a.view(-1, V_a.size()[1], all_dim) #N,C,H*W
        V_b_flat = V_b.view(-1, V_b.size()[1], all_dim) #N, C, H*W
        
        D_a_flat = D_a.view(-1, D_a.size()[1], all_dim) # N, C2, H*W
        D_b_flat = D_b.view(-1, D_b.size()[1], all_dim) # N, C2, H*W

        V_a_flat = np.concatenate((V_a_flat, D_a_flat), axis=1) # merge the rgb channels and the depth channel
        V_b_flat = np.concatenate((V_b_flat, D_b_flat), axis=1) # merge the rgb channels and the depth channel
        
        # S = B W A_transform
        encoder_future_1_flat_t = torch.transpose(V_a_flat,1,2).contiguous()  #N, H*W, C*2
        weighted_encoder_feature_1 = self.linear_e(encoder_future_1_flat_t) # weighted_encoder_feature_1 = encoder_future_1_flat_t * W, [N, H*W, C*2]
        S = torch.bmm(weighted_encoder_feature_1, V_b_flat) # S = weighted_encoder_feature_1 prod encoder_feature_2_flat, [N, H*W, H*W]

        S_row = F.softmax(S.clone(), dim = 1) # every slice along dim 1 will sum to 1, S row-wise
        S_column = F.softmax(torch.transpose(S,1,2),dim=1) # S column-wise

        Z_b = torch.bmm(V_a_flat, S_row).contiguous() #Z_b = V_a_flat prod S_row
        Z_a = torch.bmm(V_b_flat, S_column).contiguous() # Z_a = V_b_flat prod S_column
        
        input1_att = Z_a.view(-1, V_b.size()[1], fea_size[0], fea_size[1]) # [N, C*2, H, W]
        input2_att = Z_b.view(-1, V_b.size()[1], fea_size[0], fea_size[1]) # [N, C*2, H, W]
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
        #print("after upsample, tensor size:", x.size())
        x1 = self.softmax(x1)
        x2 = self.softmax(x2)

        return x1, x2, labels  #shape: [N, 1, H, W]

