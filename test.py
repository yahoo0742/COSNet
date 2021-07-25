# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import yaml
with open("config.yaml") as config_file:
    user_config = yaml.load(config_file)

#sys.path.append('/vol/graphics-solar/fengwenb/vos/cosnet/COSNet')
#print(sys.path)
import argparse
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
from dataloaders import hzfu_rgbd_loader as hzfurgbd_db
from dataloaders import PairwiseImg_test as db
#from dataloaders import StaticImg as db #采用voc dataset的数据设置格式方法
import matplotlib.pyplot as plt
import random
import timeit
from PIL import Image
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch.nn as nn
#from utils.colorize_mask import cityscapes_colorize_mask, VOCColorize
#import pydensecrf.densecrf as dcrf
#from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
from deeplab.residual_net import Bottleneck
from rgbd_segmentation_model import RGBDSegmentationModel

from torchvision.utils import save_image

from evaluation import compute_iou

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="PSPnet")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        help="voc12, cityscapes, or pascal-context")

    # GPU configuration
    parser.add_argument("--cuda", default=True, help="Run on CPU or GPU")
    parser.add_argument("--gpus", type=str, default="0",
                        help="choose gpu device.")
    parser.add_argument("--seq_name", default = 'bmx-bumps')
    parser.add_argument("--use_crf", default = 'True')
    parser.add_argument("--sample_range", default =5)
    
    return parser.parse_args()

def configure_dataset_model(args):
    if args.dataset == 'hzfurgb':
        args.data_dir = '/vol/graphics-solar/fengwenb/vos/dataset/RGBD_video_seg_dataset',
        args.batch_size = 1# 1 card: 5, 2 cards: 10 Number of images sent to the network in one step, 16 on paper
        args.maxEpoches = 15 # 1 card: 15, 2 cards: 15 epoches, equal to 30k iterations, max iterations= maxEpoches*len(train_aug)/batch_size_per_gpu'),
        args.ignore_label = 255     #The index of the label to ignore during the training
        args.input_size = '640,480' #'1920,1080' #Comma-separated string with height and width of images
        args.num_classes = 2      #Number of classes to predict (including background)
        args.img_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)       # saving model file and log record during the process of training
        args.restore_from = './pretrained/co_attention.pth' #'./your_path.pth' #resnet50-19c8e357.pth''/home/xiankai/PSPNet_PyTorch/snapshots/davis/psp_davis_0.pth' #
        args.snapshot_dir = './snapshots/hzfurgb_iteration/'          #Where to save snapshots of the model
        args.save_segimage = True
        args.seg_save_dir = "./result/test/hzfurgb"
        args.vis_save_dir = "./result/test/hzfurgb_vis"
        args.corp_size =(473, 473) #didn't see reference
        
    elif args.dataset == 'hzfurgbd': 
        args.data_dir = '/vol/graphics-solar/fengwenb/vos/dataset/RGBD_video_seg_dataset',
        args.batch_size = 1# 1 card: 5, 2 cards: 10 Number of images sent to the network in one step, 16 on paper
        args.maxEpoches = 15 # 1 card: 15, 2 cards: 15 epoches, equal to 30k iterations, max iterations= maxEpoches*len(train_aug)/batch_size_per_gpu'),
        args.ignore_label = 255     #The index of the label to ignore during the training
        args.input_size = '640,480' #'1920,1080' #Comma-separated string with height and width of images
        args.num_classes = 2      #Number of classes to predict (including background)
        args.img_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)       # saving model file and log record during the process of training
        args.restore_from = './pretrained/co_attention.pth' #'./your_path.pth' #resnet50-19c8e357.pth''/home/xiankai/PSPNet_PyTorch/snapshots/davis/psp_davis_0.pth' #
        args.snapshot_dir = './snapshots/co_attention_rgbd_hzfurgbd_29.pth/'          #Where to save snapshots of the model
        args.save_segimage = True
        args.seg_save_dir = "./result/test/hzfurgbd"
        args.vis_save_dir = "./result/test/hzfurgbd_vis"
        args.corp_size =(473, 473) #didn't see reference
        
    else:
        print("dataset error")

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
       You probably saved the model using nn.DataParallel, which stores the model in module, and now you are trying to load it 
       without DataParallel. You can either add a nn.DataParallel temporarily in your network for loading purposes, or you can 
       load the weights file, create a new ordered dict without the module prefix, and load it back 
    """
    state_dict_new = OrderedDict()
    #print(type(state_dict))
    for k, v in state_dict.items():
        print("state key: ",k)
        name = k[7:] # remove the prefix module.
        # My heart is broken, the pytorch have no ability to do with the problem.
        state_dict_new[name] = v
        if name == 'linear_e.weight':
            np.save('weight_matrix.npy',v.cpu().numpy())
    return state_dict_new

def sigmoid(inX): 
    return 1.0/(1+np.exp(-inX))#定义一个sigmoid方法，其本质就是1/(1+e^-x)

def main():
    args = get_arguments()
    print("=====> Configure dataset and model")
    configure_dataset_model(args)
    print(args)
    model = RGBDSegmentationModel(Bottleneck, [3, 4, 23, 3], num_classes=args.num_classes)
    saved_state_dict = torch.load(args.restore_from, map_location=lambda storage, loc: storage)
    #print(saved_state_dict.keys())
    #model.load_state_dict({k.replace('pspmodule.',''):v for k,v in torch.load(args.restore_from)['state_dict'].items()})
    model.load_state_dict( convert_state_dict(saved_state_dict["model"]) ) #convert_state_dict(saved_state_dict["model"])

    model.eval()
    model.cuda()
    if args.dataset == 'davis':  #for davis 2016
        db_test = db.PairwiseImg(train=False, inputRes=(854,480), db_root_dir=args.data_dir,  transform=None, seq_name = None, sample_range = args.sample_range) #db_root_dir() --> '/path/to/DAVIS-2016' train path
        testloader = data.DataLoader(db_test, batch_size= 10, shuffle=False, num_workers=0)
        #voc_colorize = VOCColorize()
    elif args.dataset == 'hzfurgb':
        db_test = hzfurgbd_db.HzFuRGBDVideos(args.data_dir, sample_range=args.sample_range)
        db_test.set_for_test()
        testloader = data.DataLoader(db_test, batch_size= 10, shuffle=False, num_workers=0)
    elif args.dataset == 'hzfurgbd':
        db_test = hzfurgbd_db.HzFuRGBDVideos(args.data_dir, sample_range=args.sample_range)
        db_test.set_for_test()
        testloader = data.DataLoader(db_test, batch_size= 10, shuffle=False, num_workers=0)
    else:
        print("dataset error")

    data_list = []

    if args.save_segimage:
        if not os.path.exists(args.seg_save_dir) and not os.path.exists(args.vis_save_dir):
            os.makedirs(args.seg_save_dir)
            os.makedirs(args.vis_save_dir)
    print("======> test set size:", len(testloader))
    my_index = 0
    old_temp=''
    for index, batch in enumerate(testloader):
        print('%d processd'%(index))
        # current_img, current_depth, current_img_gt, match_img, match_depth, match_img_gt
        target = batch['target']
        target_depth = batch['target_depth']
        frame_index = batch['frame_index']
        #search = batch['search']
        temp = batch['seq_name']

        args.seq_name=temp[0]
        # print(args.seq_name)
        if old_temp==args.seq_name:
            my_index = my_index+1
        else:
            my_index = 0

        seq_name = ""

        output_sum = 0 
        for i in range(0,args.sample_range):  
            search = batch['search'+'_'+str(i)]
            search_depth = batch['search_depth'+'_'+str(i)]
            search_im = search
            #print(search_im.size())
            with torch.no_grad():
                output = model(Variable(target).cuda(),Variable(search_im).cuda(), Variable(target_depth).cuda(), Variable(search_depth).cuda())
                #print(output[0]) # output有两个
                # output_sum = output_sum + output[0].data[0,0].cpu().numpy() #分割那个分支的结果
                output_sum = output_sum + output[0].data.cpu().numpy() #分割那个分支的结果^M
                #np.save('infer'+str(i)+'.npy',output1)
                #output2 = output[1].data[0, 0].cpu().numpy() #interp'
        
        output1 = output_sum/args.sample_range
     
        # first_image = np.array(Image.open(args.data_dir+'/JPEGImages/480p/blackswan/00000.jpg'))
        original_shape = (480, 640) #first_image.shape 
        outputarray = np.array(output1)

        output2 = []
        for idx in range(len(output1)):
            img = output1[idx, 0]
            img = cv2.resize(img, (original_shape[1],original_shape[0]))
            output2.append(img)
        output1 = np.array(output2)
        # output1 = cv2.resize(output1, (original_shape[1],original_shape[0]))

        masks_data = (output1 * 255).astype(np.uint8)
        masks = []
        for idx in range(len(masks_data)):
            x = masks_data[idx]
            iou = compute_iou(x, np.array(batch['target_gt'][idx]))
            mask = Image.fromarray(x, mode='L')
            masks.append(mask)

        # mask = (output1*255).astype(np.uint8)
        # #print(mask.shape[0])
        # mask = Image.fromarray(mask)

        if args.dataset == 'davis':
            
            save_dir_res = os.path.join(args.seg_save_dir, 'Results', args.seq_name)
            old_temp=args.seq_name
            if not os.path.exists(save_dir_res):
                os.makedirs(save_dir_res)
            if args.save_segimage:   
                my_index1 = str(my_index).zfill(5)
                seg_filename = os.path.join(save_dir_res, '{}.png'.format(my_index1))
                #color_file = Image.fromarray(voc_colorize(output).transpose(1, 2, 0), 'RGB')
                mask.save(seg_filename)
                #np.concatenate((torch.zeros(1, 473, 473), mask, torch.zeros(1, 512, 512)),axis = 0)
                #save_image(output1 * 0.8 + target.data, args.vis_save_dir, normalize=True)
        elif args.dataset == 'hzfurgbd':
            if args.save_segimage:
                save_dir_res = os.path.join(args.seg_save_dir, 'Results', seq_name)
                if not os.path.exists(save_dir_res):
                    os.makedirs(save_dir_res)
                for idx in range(len(masks)):
                    mask = masks[idx]
                    seg_filename = os.path.join(save_dir_res, '{}.png'.format(frame_index[idx]))
                    mask.save(seg_filename)

        else:
            print("dataset error")
    

if __name__ == '__main__':
    main()
