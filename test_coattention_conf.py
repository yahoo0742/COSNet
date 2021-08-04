# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 17:53:20 2018

@author: carri
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

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
from deeplab.siamese_model_conf import CoattentionNet
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
    if args.dataset == 'voc12':
        args.data_dir ='/home/wty/AllDataSet/VOC2012'  #Path to the directory containing the PASCAL VOC dataset
        args.data_list = './dataset/list/VOC2012/test.txt'  #Path to the file listing the images in the dataset
        args.img_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32) 
        #RBG mean, first subtract mean and then change to BGR
        args.ignore_label = 255   #The index of the label to ignore during the training
        args.num_classes = 21  #Number of classes to predict (including background)
        args.restore_from = './snapshots/voc12/psp_voc12_14.pth'  #Where restore model parameters from
        args.save_segimage = True
        args.seg_save_dir = "./result/test/VOC2012"
        args.corp_size =(505, 505)
        
    elif args.dataset == 'davis': 
        args.batch_size = 10# 1 card: 5, 2 cards: 10 Number of images sent to the network in one step, 16 on paper
        args.maxEpoches = 15 # 1 card: 15, 2 cards: 15 epoches, equal to 30k iterations, max iterations= maxEpoches*len(train_aug)/batch_size_per_gpu'),
        args.data_dir = '/vol/graphics-solar/fengwenb/vos/dataset/DAVIS'  #/DAVIS-2016'   # 37572 image pairs
        args.data_list = '/vol/graphics-solar/fengwenb/vos/dataset/DAVIS/ImageSets/480p/val.txt' #'your_path/DAVIS-2016/test_seqs.txt'  # Path to the file listing the images in the dataset
        args.ignore_label = 255     #The index of the label to ignore during the training
        args.input_size = '854,480' # '854,480' W, H #'1920,1080' #Comma-separated string with height and width of images
        args.desired_HW = '120,214' # H, W
        args.num_classes = 2      #Number of classes to predict (including background)
        args.img_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)       # saving model file and log record during the process of training
        args.restore_from = './pretrained/co_attention_davis_29.pth' #'./pretrained/co_attention.pth' #'./your_path.pth' #resnet50-19c8e357.pth''/home/xiankai/PSPNet_PyTorch/snapshots/davis/psp_davis_0.pth' #
        args.snapshot_dir = './snapshots/davis_iteration/'          #Where to save snapshots of the model
        args.save_segimage = True
        args.seg_save_dir = "./result/test/davis_iteration_conf"
        args.vis_save_dir = "./result/test/davis_vis"
        args.corp_size =(473, 473) #didn't see reference

    elif args.dataset == 'hzfuRGB': 
        args.batch_size = 1# 1 card: 5, 2 cards: 10 Number of images sent to the network in one step, 16 on paper
        args.maxEpoches = 15 # 1 card: 15, 2 cards: 15 epoches, equal to 30k iterations, max iterations= maxEpoches*len(train_aug)/batch_size_per_gpu'),
        args.data_dir = '/vol/graphics-solar/fengwenb/vos/dataset/DAVIS'  #/DAVIS-2016'   # 37572 image pairs
        args.data_list = '/vol/graphics-solar/fengwenb/vos/dataset/DAVIS/ImageSets/480p/val.txt' #'your_path/DAVIS-2016/test_seqs.txt'  # Path to the file listing the images in the dataset
        args.ignore_label = 255     #The index of the label to ignore during the training
        args.input_size = '640,480' #'1920,1080' W, H #Comma-separated string with height and width of images
        args.desired_HW = '120,160' #H, W
        args.num_classes = 2      #Number of classes to predict (including background)
        args.img_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)       # saving model file and log record during the process of training
        args.restore_from = './pretrained/co_attention.pth' #'./your_path.pth' #resnet50-19c8e357.pth''/home/xiankai/PSPNet_PyTorch/snapshots/davis/psp_davis_0.pth' #
        args.snapshot_dir = './snapshots/davis_iteration/'          #Where to save snapshots of the model
        args.save_segimage = True
        args.seg_save_dir = "./result/test/hzfuRGB"
        args.vis_save_dir = "./result/test/hzfuRGB_vis"
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
        #print(k)
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
    h, w = map(int, args.desired_HW.split(','))
    args.desired_HW = (h, w)
    w, h = map(int, args.input_size.split(','))
    args.input_size = (w,h)
    print(args)
    if args.cuda:
        print("====> Use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    model = CoattentionNet(num_classes=args.num_classes)
    
    saved_state_dict = torch.load(args.restore_from, map_location=lambda storage, loc: storage)
    #print(saved_state_dict.keys())
    #model.load_state_dict({k.replace('pspmodule.',''):v for k,v in torch.load(args.restore_from)['state_dict'].items()})
    model.load_state_dict( convert_state_dict(saved_state_dict["model"]) ) #convert_state_dict(saved_state_dict["model"])

    model.eval()
    model.cuda()
    if args.dataset == 'voc12':
        testloader = data.DataLoader(VOCDataTestSet(args.data_dir, args.data_list, crop_size=(505, 505),mean= args.img_mean), 
                                    batch_size=1, shuffle=False, pin_memory=True)
        interp = nn.Upsample(size=(505, 505), mode='bilinear')
        voc_colorize = VOCColorize()
    
    elif args.dataset == 'hzfuRGB':
        db_test = hzfurgbd_db.HzFuRGBDVideos('/vol/graphics-solar/fengwenb/vos/dataset/RGBD_video_seg_dataset', sample_range=args.sample_range, desired_HW=args.desired_HW)
        db_test.set_for_test()
        testloader = data.DataLoader(db_test, batch_size= 10, shuffle=False, num_workers=0)
    elif args.dataset == 'davis':  #for davis 2016
        db_test = db.PairwiseImg(train=False, desired_HW=args.desired_HW, db_root_dir=args.data_dir,  transform=None, seq_name = None, sample_range = args.sample_range) #db_root_dir() --> '/path/to/DAVIS-2016' train path
        testloader = data.DataLoader(db_test, batch_size= 10, shuffle=False, num_workers=0)
        #voc_colorize = VOCColorize()
    else:
        print("dataset error")

    data_list = []

    if args.save_segimage:
        if not os.path.exists(args.seg_save_dir) and not os.path.exists(args.vis_save_dir):
            os.makedirs(args.seg_save_dir)
            os.makedirs(args.vis_save_dir)
    my_index = 0
    old_temp=''
    for index, batch in enumerate(testloader):
        print('%d processd'%(index))
        target = batch['target']
        frame_index = batch['frame_index']
        #search = batch['search']
        temp = batch['seq_name']
        args.seq_name=temp[0]
        # print(args.seq_name)
        if old_temp==args.seq_name:
            my_index = my_index+1
        else:
            my_index = 0
        output_sum = 0 
        for i in range(0,args.sample_range):  
            search = batch['search'+'_'+str(i)]
            search_im = search
            #print(search_im.size())
            with torch.no_grad():
                output = model(Variable(target).cuda(),Variable(search_im, volatile=True).cuda())
                #print(output[0]) # output有两个
                # output_sum = output_sum + output[0].data[0,0].cpu().numpy() #分割那个分支的结果
                output_sum = output_sum + output[0].data.cpu().numpy() #分割那个分支的结果^M
                #np.save('infer'+str(i)+'.npy',output1)
                #output2 = output[1].data[0, 0].cpu().numpy() #interp'
        output1 = output_sum/args.sample_range
     
        # first_image = np.array(Image.open(args.data_dir+'/JPEGImages/480p/blackswan/00000.jpg'))
        outputarray = np.array(output1)
        output2 = []
        for idx in range(len(output1)):
            img = output1[idx, 0]
            img = cv2.resize(img, (args.desired_HW[1], args.desired_HW[0])) # (w, h)
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
        #print(mask.shape[0])
        # mask = Image.fromarray(mask)

        if args.dataset == 'voc12':
            print(output.shape)
            print(size)
            output = output[:,:size[0],:size[1]]
            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            if args.save_segimage:
                seg_filename = os.path.join(args.seg_save_dir, '{}.png'.format(name[0]))
                color_file = Image.fromarray(voc_colorize(output).transpose(1, 2, 0), 'RGB')
                color_file.save(seg_filename)
                
        elif args.dataset == 'davis':
            
            save_dir_res = os.path.join(args.seg_save_dir, 'Results', args.seq_name)
            old_temp=args.seq_name
            if not os.path.exists(save_dir_res):
                os.makedirs(save_dir_res)
            if args.save_segimage:   
                my_index1 = str(my_index).zfill(5)
                for idx in range(len(masks)):
                    mask = masks[idx]
                    seg_filename = os.path.join(save_dir_res, '{}.png'.format(frame_index[idx]))
                    mask.save(seg_filename)
                # seg_filename = os.path.join(save_dir_res, '{}.png'.format(my_index1))
                #color_file = Image.fromarray(voc_colorize(output).transpose(1, 2, 0), 'RGB')
                # mask.save(seg_filename)
                #np.concatenate((torch.zeros(1, 473, 473), mask, torch.zeros(1, 512, 512)),axis = 0)
                #save_image(output1 * 0.8 + target.data, args.vis_save_dir, normalize=True)
        else:
            print("dataset error")
    

if __name__ == '__main__':
    main()
