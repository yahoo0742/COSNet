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
from dataloaders import sbm_rgbd_loader as sbmrgbd_db
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
from rgbd_segmentation_RAA import RGBDSegmentation_RAA
from rgbd_segmentation_model import RGBDSegmentationModel
from deeplab.siamese_model_conf import CoattentionNet #original coattention model 
from deeplab.siamese_model import CoattentionSiameseNet #refactored coattention model


from torchsummary import summary
from torchvision.utils import save_image

from evaluation import compute_iou
import datetime


log_section_start = "##=="
log_section_end = "==##"

timenow = datetime.datetime.now()
ymd_hms = timenow.strftime("%Y%m%d_%H%M%S")


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="RGBDCoAttention")
    parser.add_argument("--dataset", type=str, default='hzfurgbd',
                        help="hzfurgb, hzfurgbd, sbmrgbd, or davis")

    # GPU configuration
    parser.add_argument("--cuda", default=True, help="Run on CPU or GPU")
    parser.add_argument("--gpus", type=str, default="0",
                        help="choose gpu device.")
    parser.add_argument("--seq_name", default = 'bmx-bumps')
    parser.add_argument("--use_crf", default = 'True')
    parser.add_argument("--save_seg_img", default = 'True')
    parser.add_argument("--sample_range", default =5)
    parser.add_argument("--epoches", default=0)
    parser.add_argument("--batch_size", default=0)
    parser.add_argument("--model", default="raa", help="ori, retrain, ref, add, padd, conv_add, or coc")

    return parser.parse_args()


def config(args):
    if args.dataset == 'hzfud':
        if not args.batch_size:
            args.batch_size = 1# 1 card: 5, 2 cards: 10 Number of images sent to the network in one step, 16 on paper
        if not args.epoches:
            args.epoches = 15 # 1 card: 15, 2 cards: 15 epoches, equal to 30k iterations, max iterations= epoches*len(train_aug)/batch_size_per_gpu'),
        args.ignore_label = 255     #The index of the label to ignore during the training
        args.num_classes = 2      #Number of classes to predict (including background)
        args.img_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)       # saving model file and log record during the process of training
        #args.restore_from = './pretrained/co_attention.pth' #'./your_path.pth' #resnet50-19c8e357.pth''/home/xiankai/PSPNet_PyTorch/snapshots/davis/psp_davis_0.pth' #

    elif args.dataset == 'hzfurgb':
        if not args.batch_size:
            args.batch_size = 1# 1 card: 5, 2 cards: 10 Number of images sent to the network in one step, 16 on paper
        if not args.epoches:
            args.epoches = 15 # 1 card: 15, 2 cards: 15 epoches, equal to 30k iterations, max iterations= epoches*len(train_aug)/batch_size_per_gpu'),
        args.ignore_label = 255     #The index of the label to ignore during the training
        args.num_classes = 2      #Number of classes to predict (including background)
        args.img_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)       # saving model file and log record during the process of training
        #args.restore_from = './pretrained/co_attention.pth' #'./your_path.pth' #resnet50-19c8e357.pth''/home/xiankai/PSPNet_PyTorch/snapshots/davis/psp_davis_0.pth' #

    elif args.dataset == 'hzfurgbd': 
        if not args.batch_size:
            args.batch_size = 1# 1 card: 5, 2 cards: 10 Number of images sent to the network in one step, 16 on paper
        if not args.epoches:
            args.epoches = 15 # 1 card: 15, 2 cards: 15 epoches, equal to 30k iterations, max iterations= epoches*len(train_aug)/batch_size_per_gpu'),

        args.ignore_label = 255     #The index of the label to ignore during the training
        args.num_classes = 2      #Number of classes to predict (including background)
        args.img_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)       # saving model file and log record during the process of training
        #args.restore_from = './snapshots/co_attention_rgbd_hzfurgbd_29.pth' #'./your_path.pth' #resnet50-19c8e357.pth''/home/xiankai/PSPNet_PyTorch/snapshots/davis/psp_davis_0.pth' #

    elif args.dataset == 'sbmrgbd': 
        if not args.batch_size:
            args.batch_size = 1# 1 card: 5, 2 cards: 10 Number of images sent to the network in one step, 16 on paper
        if not args.epoches:
            args.epoches = 15 # 1 card: 15, 2 cards: 15 epoches, equal to 30k iterations, max iterations= epoches*len(train_aug)/batch_size_per_gpu'),

        args.num_classes = 2      #Number of classes to predict (including background)
        args.img_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)       # saving model file and log record during the process of training
        #args.restore_from = './snapshots/co_attention_rgbd_hzfurgbd_29.pth' #'./your_path.pth' #resnet50-19c8e357.pth''/home/xiankai/PSPNet_PyTorch/snapshots/davis/psp_davis_0.pth' #

    else:
        print("dataset error")

    args.data_path = user_config['test']['dataset'][args.dataset]['data_path'] #'/vol/graphics-solar/fengwenb/vos/dataset/RGBD_video_seg_dataset'
    args.sample_range = user_config['test']['dataset'][args.dataset]['sample_range']

    h, w = map(int, user_config['test']['dataset'][args.dataset]['image_HW_4_model'].split(','))
    args.image_HW_4_model = (h, w)
    w, h = map(int, user_config['test']['dataset'][args.dataset]['output_WH'].split(','))
    args.output_WH = (w, h)


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
        if k.startswith("module."):
            name = k[k.index(".")+1:] # k is like module.encoder.backbone.layer2.xxxx
        else:
            name = k
        #name = k # k[7:] # remove the prefix module.
        # My heart is broken, the pytorch have no ability to do with the problem.
        state_dict_new[name] = v
        if name == 'linear_e.weight':
            np.save('weight_matrix.npy',v.cpu().numpy())
    return state_dict_new


def sigmoid(inX): 
    return 1.0/(1+np.exp(-inX))#定义一个sigmoid方法，其本质就是1/(1+e^-x)


def main():
    args = get_arguments()
    config(args)
    print(args)

    if args.cuda:
        print("====> Use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    args.full_model_name = ""
    if args.model == "ori" or args.model == "original_coattention_rgb":
        model = CoattentionNet(num_classes=2)
        args.full_model_name = "original_coattention_rgb"
    elif args.model == "retrain" or args.model == "original_coattention_rgb_retrained":
        model = CoattentionNet(num_classes=2)
        args.full_model_name = "original_coattention_rgb_retrained"
    elif args.model == "ref" or args.model == "refactored_coattention_rgb":
        model = CoattentionSiameseNet(Bottleneck, 3, [3, 4, 23, 3], 1)
        args.full_model_name = "refactored_coattention_rgb"
    elif args.model == "raa" or args.model == "resnet_aspp_add":
        model = RGBDSegmentation_RAA(Bottleneck, [3, 4, 23, 3], [3, 4, 6, 3], num_classes=1)
        args.full_model_name = "resnet_aspp_add"
    elif args.model == "add" or args.model == "added_depth_rgbd":
        model = RGBDSegmentationModel(Bottleneck, [3, 4, 23, 3],  [3, 4, 6, 3], 1, approach_for_depth="add")
        args.full_model_name = "added_depth_rgbd"
    elif args.model == "conc1" or args.model == "concatenated_depth_rgbd":
        model = RGBDSegmentationModel(Bottleneck, [3, 4, 23, 3], [3, 4, 6, 3], 1, approach_for_depth="conc1")
        args.full_model_name = "concatenated_depth_rgbd"
    elif args.model == "conc2" or args.model == "concatenated_depth_rgbd2":
        model = RGBDSegmentationModel(Bottleneck, [3, 4, 23, 3], [3, 4, 6, 3], num_classes=1, approach_for_depth="conc2")
        args.full_model_name = "concatenated_depth_rgbd2"
    elif args.model == "padd" or args.model == "post_added_depth_rgbd":
        model = RGBDSegmentationModel(Bottleneck, [3, 4, 23, 3], None, num_classes=1, approach_for_depth="padd")
        args.full_model_name = "post_added_depth_rgbd"
    elif args.model == "conv_add" or args.model == "convs_depth_addition":
        model = RGBDSegmentationModel(Bottleneck, [3, 4, 23, 3], None, num_classes=1, approach_for_depth="conv_add")
        args.full_model_name = "convs_depth_addition"
    else:
        print("Invalid model name!")
        return

    args.result_dir = os.path.join(".", "vos_test_results", args.dataset, args.full_model_name, ymd_hms)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    logFileName = os.path.join(args.result_dir, args.dataset+"__"+args.full_model_name+"_"+ymd_hms+"_test_log.txt")
    print("Logs will be writen in "+logFileName +" and the test results will be in "+args.result_dir)
    if os.path.isfile(logFileName):
        logger = open(logFileName, 'a')
    else:
        logger = open(logFileName, 'w')

    args.pretrained_params = user_config['test']['model'][args.full_model_name]['pretrained_params']
    logger.write(log_section_start+str(args)+log_section_end+"\n")
    logger.flush()

    saved_state_dict = torch.load(args.pretrained_params, map_location=lambda storage, loc: storage)
    #print(saved_state_dict.keys())
    #model.load_state_dict({k.replace('pspmodule.',''):v for k,v in torch.load(args.restore_from)['state_dict'].items()})
    model.load_state_dict( convert_state_dict(saved_state_dict["model"]) ) #convert_state_dict(saved_state_dict["model"])

    model.eval()
    if args.full_model_name == "resnet_aspp_add"
        summary(model, [(3,480,640),(3,480,640),(1,480,640), (1,480,640)]) #summary the model
    elif args.full_model_name == "added_depth_rgbd" or args.full_model_name == "post_added_depth_rgbd" or args.full_model_name == "concatenated_depth_rgbd" or args.full_model_name == "concatenated_depth_rgbd2" or args.full_model_name == "convs_depth_addition":
        summary(model, [(3,480,640),(3,480,640),(1,480,640)]) #summary the model
    else:
        summary(model, [(3,480,640),(3,480,640)])

    model.cuda()

    if args.dataset == 'davis':  #for davis 2016
        db_test = db.PairwiseImg(train=False, output_HW=(854,480), db_root_dir=args.data_path,  transform=None, seq_name = None, sample_range = args.sample_range) #db_root_dir() --> '/path/to/DAVIS-2016' train path
        testloader = data.DataLoader(db_test, batch_size= args.batch_size, shuffle=False, num_workers=0)
        #voc_colorize = VOCColorize()
    elif args.dataset == 'hzfud':
        db_test = hzfurgbd_db.HzFuRGBDVideos(dataset_root=args.data_path, output_HW=args.image_HW_4_model, sample_range=args.sample_range, channels_for_target_frame='dt', channels_for_counterpart_frame='d', subset_percentage=1, subset=user_config["test"]["dataset"]["hzfurgb"]["subset"], for_training=False, batch_size=args.batch_size)
        testloader = data.DataLoader(db_test, batch_size= args.batch_size, shuffle=True, num_workers=0)
    elif args.dataset == 'hzfurgb':
        db_test = hzfurgbd_db.HzFuRGBDVideos(dataset_root=args.data_path, output_HW=args.image_HW_4_model, sample_range=args.sample_range, channels_for_target_frame='rgbt', channels_for_counterpart_frame='rgb', subset_percentage=1, subset=user_config["test"]["dataset"]["hzfurgb"]["subset"], for_training=False, batch_size=args.batch_size)
        testloader = data.DataLoader(db_test, batch_size= args.batch_size, shuffle=True, num_workers=0)
    elif args.dataset == 'hzfurgbd':
        db_test = hzfurgbd_db.HzFuRGBDVideos(dataset_root=args.data_path, output_HW=args.image_HW_4_model, sample_range=args.sample_range, channels_for_target_frame='rgbdt', channels_for_counterpart_frame='rgbd',  subset_percentage=1, subset=user_config["test"]["dataset"]["hzfurgbd"]["subset"], for_training=False, batch_size=args.batch_size)
        testloader = data.DataLoader(db_test, batch_size= args.batch_size, shuffle=True, num_workers=0)
    elif args.dataset == 'sbmrgbd':
        db_test = sbmrgbd_db.sbm_rgbd(dataset_root=args.data_path, output_HW=args.image_HW_4_model, sample_range=args.sample_range, channels_for_target_frame='rgbdt', channels_for_counterpart_frame='rgbd',  subset_percentage=1, subset=user_config["test"]["dataset"]["sbmrgbd"]["subset"], for_training=False, batch_size=args.batch_size)
        testloader = data.DataLoader(db_test, batch_size= args.batch_size, shuffle=True, num_workers=0)
    else:
        print("dataset error")

    data_list = []

    if args.save_seg_img:
        args.output_img_dir = os.path.join(args.result_dir, "obj_seg_imgs")
        if not os.path.exists(args.output_img_dir):
            os.makedirs(args.output_img_dir)
        
    print("======> test set size:", len(testloader))

    my_index = 0
    old_temp=''

    iou_result = 0
    iou_counter = 0

    ct = 0
    max_epoch = 1
    while ct < max_epoch:
        ct = ct + 1
        for index, batch in enumerate(testloader):
            print('%d processd'%(index))
            # current_img, current_depth, current_img_gt, match_img, match_depth, match_img_gt
            target = batch['target']
            target_depth = batch['target_depth']
            frame_index = batch['frame_index']
            seqs_name = batch['seq_name']

            output_sum = 0 
            for i in range(0,args.sample_range):  
                search_img = batch['search'+'_'+str(i)]
                search_depth = batch['search_'+str(i)+'_depth']
                #print(search_img.size())
                with torch.no_grad():
                    if args.full_model_name == "resnet_aspp_add"
                        output = model(Variable(target).cuda(),Variable(search_img).cuda(), Variable(target_depth).cuda(), Variable(search_depth).cuda())
                    elif args.full_model_name == "added_depth_rgbd" or args.full_model_name == "post_added_depth_rgbd" or args.full_model_name == "concatenated_depth_rgbd" or args.full_model_name == "concatenated_depth_rgbd2" or args.full_model_name == "convs_depth_addition":
                        output = model(Variable(target).cuda(),Variable(search_img).cuda(), Variable(target_depth).cuda())
                    else:
                        output = model(Variable(target).cuda(),Variable(search_img).cuda())

                    #print(output[0]) # output有两个
                    # output_sum = output_sum + output[0].data[0,0].cpu().numpy() #分割那个分支的结果
                    output_sum = output_sum + output[0].data.cpu().numpy() #分割那个分支的结果^M
                    #np.save('infer'+str(i)+'.npy',output1)
                    #output2 = output[1].data[0, 0].cpu().numpy() #interp'
            
            output1 = output_sum/args.sample_range
            outputarray = np.array(output1)

            # resize
            output2 = []
            for idx in range(len(output1)):
                img = output1[idx, 0]
                img = cv2.resize(img, args.output_WH)
                output2.append(img)
            output1 = np.array(output2)

            # calc IOU and generate the final images
            masks_data_uint8 = (output1 * 255).astype(np.uint8)
            masks = []
            for idx in range(len(masks_data_uint8)):
                x = masks_data_uint8[idx]
                iou = compute_iou(x, np.array(batch['target_gt'][idx]))
                logger.write(log_section_start+" seq: "+ seqs_name[idx]+" frame: "+frame_index[idx]+" IOU: "+str(iou)+log_section_end+"\n")
                iou_result = iou + iou_result
                iou_counter = iou_counter + 1
                mask = Image.fromarray(x, mode='L')
                masks.append(mask)

            # mask = (output1*255).astype(np.uint8)
            # #print(mask.shape[0])
            # mask = Image.fromarray(mask)

            if args.output_img_dir:
                
                for idx in range(len(masks)):
                    mask = masks[idx]
                    save_dir = os.path.join(args.output_img_dir, seqs_name[idx])
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    seg_filename = os.path.join(save_dir, '{}.png'.format(frame_index[idx]))
                    mask.save(seg_filename)

    iou_result = iou_result/iou_counter
    logger.write(log_section_start+" final IOU: "+ str(iou_result) +log_section_end+"\n")
    logger.flush()


if __name__ == '__main__':
    main()
