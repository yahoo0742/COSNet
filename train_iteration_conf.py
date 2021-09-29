# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 10:52:26 2018

@author: carri
"""
#区别于deeplab_co_attention_concat在于采用了新的model（siamese_model_concat_new）来train

# Editing this file is too tricky. Load configurations from the yaml file, so that when we need to customize something, we do not need to edit this file
import yaml
with open("config.yaml") as config_file:
    user_config = yaml.load(config_file)

import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils import data
import pickle
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
#from utils.balanced_BCE import class_balanced_cross_entropy_loss
import os.path as osp
#from psp.model import PSPNet
#from dataloaders import davis_2016 as db
from dataloaders import PairwiseImg_video as db #采用voc dataset的数据设置格式方法
import matplotlib.pyplot as plt
import random
import timeit
#from psp.model1 import CoattentionNet  #基于pspnet搭建的co-attention 模型
from deeplab.siamese_model_conf import CoattentionNet #siame_model 是直接将attend的model之后的结果输出
#from deeplab.utils import get_1x_lr_params, get_10x_lr_params#, adjust_learning_rate #, loss_calc
from deeplab.residual_net import Bottleneck
from deeplab.siamese_model import CoattentionSiameseNet
import gc
import datetime

start = timeit.default_timer()
timenow = datetime.datetime.now()
ymd_hms = timenow.strftime("%Y%m%d_%H%M%S")

def logMem(logger, prefix):
    total = torch.cuda.get_device_properties(None).total_memory
    mem_alloc = torch.cuda.memory_allocated()
    mem_cache = torch.cuda.memory_cached()
    msg = prefix + " mem_alloc: "+str(mem_alloc)+"  mem_cache: "+str(mem_cache)+"  total: "+str(total)
    print(msg)
    logger.write(msg)

def plot2d(x, y, xlabel=None, ylabel=None, filenameForSave=None):
    plt.plot(x, y)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    
    if filenameForSave:
        plt.savefig(filenameForSave+".png")
    plt.show()


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="PSPnet Network")

    # optimatization configuration
    parser.add_argument("--is-training", action="store_true", 
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default= 0.00025, 
                        help="Base learning rate for training with polynomial decay.") #0.001
    parser.add_argument("--weight-decay", type=float, default= 0.0005, 
                        help="Regularization parameter for L2-loss.")  # 0.0005
    parser.add_argument("--momentum", type=float, default= 0.9, 
                        help="Momentum component of the optimiser.")
    parser.add_argument("--power", type=float, default= 0.9, 
                        help="Decay parameter to compute the learning rate.")
    # dataset information
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        help="voc12, cityscapes, or pascal-context.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")

    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--random-seed", type=int, default= 1234,
                        help="Random seed to have reproducible results.")
    parser.add_argument('--logFile', default='log.txt', 
                        help='File that stores the training and validation logs')
    # GPU configuration
    parser.add_argument("--cuda", default=True, help="Run on CPU or GPU")
    parser.add_argument("--gpus", type=str, default="3", help="choose gpu device.") #使用3号GPU
    parser.add_argument("--sample_range", type=int, default=5, help="The number of frames sampled to compare with the target frame.")
    parser.add_argument("--model", type=str, default='ori', help="ori, ref")


    return parser.parse_args()

args = get_arguments()


def configure_dataset_init_model(args):
    if args.dataset == 'voc12':

        args.batch_size = 10# 1 card: 5, 2 cards: 10 Number of images sent to the network in one step, 16 on paper
        args.maxEpoches = 15 # 1 card: 15, 2 cards: 15 epoches, equal to 30k iterations, max iterations= maxEpoches*len(train_aug)/batch_size_per_gpu'),
        args.data_dir = '/home/wty/AllDataSet/VOC2012'   # Path to the directory containing the PASCAL VOC dataset
        args.data_list = './dataset/list/VOC2012/train_aug.txt'  # Path to the file listing the images in the dataset
        args.ignore_label = 255     #The index of the label to ignore during the training
        args.input_size = '473,473' #Comma-separated string with height and width of images
        args.num_classes = 21      #Number of classes to predict (including background)

        args.img_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
        # saving model file and log record during the process of training

        #Where restore model pretrained on other dataset, such as COCO.")
        args.restore_from = './pretrained/MS_DeepLab_resnet_pretrained_COCO_init.pth'
        args.snapshot_dir = './snapshots/voc12/'          #Where to save snapshots of the model
        args.resume = './snapshots/voc12/psp_voc12_3.pth' #checkpoint log file, helping recovering training
        
    elif args.dataset == 'davis': 
        args.batch_size = user_config["train"]["dataset"]["davis"]["batch_size"]# 1 card: 5, 2 cards: 10 Number of images sent to the network in one step, 16 on paper
        args.maxEpoches = user_config["train"]["dataset"]["davis"]["max_epoches"] # 1 card: 15, 2 cards: 15 epoches, equal to 30k iterations, max iterations= maxEpoches*len(train_aug)/batch_size_per_gpu'),
        args.data_dir = user_config["train"]["dataset"]["davis"]["data_path"]   # 37572 image pairs
        args.img_dir = user_config["train"]["saliency_dataset"] #'/vol/graphics-solar/fengwenb/vos/saliency_dataset'
        args.data_list = './dataset/list/VOC2012/train_aug.txt'  # Path to the file listing the images in the dataset
        args.ignore_label = user_config["train"]["dataset"]["davis"]["ignore_label"]     #The index of the label to ignore during the training
        args.input_size = user_config["train"]["dataset"]["davis"]["input_size"] #'854,480' W, H #Comma-separated string with height and width of images
        args.output_HW = user_config["train"]["dataset"]["davis"]['output_HW']
        args.num_classes = user_config["train"]["dataset"]["davis"]["num_classes"]      #Number of classes to predict (including background)
        args.img_mean = np.array(user_config["train"]["dataset"]["davis"]["img_mean"], dtype=np.float32)       # saving model file and log record during the process of training
        #Where restore model pretrained on other dataset, such as COCO.")
        args.restore_from = user_config["train"]["model"]["original_coattention_rgb_retrained"]["initial_params"] #"./pretrained/deep_labv3/deeplab_davis_12_0.pth"
        #args.restore_from = './pretrained/deep_labv3/deeplab_davis_12_0.pth' #resnet50-19c8e357.pth''/home/xiankai/PSPNet_PyTorch/snapshots/davis/psp_davis_0.pth' #
        args.snapshot_dir = user_config["train"]["dataset"]["davis"]["snapshot_output_path"] #'./snapshots/davis_iteration_conf/'          #Where to save snapshots of the model
        args.resume = user_config["train"]["dataset"]["davis"]["checkpoint_file"] #'./snapshots/davis/co_attention_davis_124.pth' #checkpoint log file, helping recovering training
		
    elif args.dataset == 'cityscapes':
        args.batch_size = 8   #Number of images sent to the network in one step, batch_size/num_GPU=2
        args.maxEpoches = 60 #epoch nums, 60 epoches is equal to 90k iterations, max iterations= maxEpoches*len(train)/batch_size')
        # 60x2975/2=89250 ~= 90k, single_GPU_batch_size=2
        args.data_dir = '/home/wty/AllDataSet/CityScapes'   # Path to the directory containing the PASCAL VOC dataset
        args.data_list = './dataset/list/Cityscapes/cityscapes_train_list.txt'  # Path to the file listing the images in the dataset
        args.ignore_label = 255     #The index of the label to ignore during the training
        args.input_size = '720,720' #Comma-separated string with height and width of images
        args.num_classes = 19      #Number of classes to predict (including background)

        args.img_mean = np.array((73.15835921, 82.90891754, 72.39239876), dtype=np.float32)
        # saving model file and log record during the process of training

        #Where restore model pretrained on other dataset, such as coarse cityscapes
        args.restore_from = './pretrained/resnet101_pretrained_for_cityscapes.pth'
        args.snapshot_dir = './snapshots/cityscapes/'          #Where to save snapshots of the model
        args.resume = './snapshots/cityscapes/psp_cityscapes_12_3.pth' #checkpoint log file, helping recovering training
       
    else:
        print("dataset error")

def adjust_learning_rate(optimizer, i_iter, epoch, max_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    
    lr = lr_poly(args.learning_rate, i_iter, max_iter, args.power, epoch)
    optimizer.param_groups[0]['lr'] = lr
    if i_iter%3 ==0:
        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = 0
    else:
        optimizer.param_groups[0]['lr'] = 0.01*lr
        optimizer.param_groups[1]['lr'] = lr * 10
        
    return lr

def calc_loss_BCE(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    labels = torch.ge(label, 0.5).float()
#    
    label_size = label.size() # N x C x H x W
    #print(batch_size)
    num_labels_pos = torch.sum(labels) # how many entries are labeled GE than 0.5
#    
    total_label_entries =  label_size[0]* label_size[2] * label_size[3]
    positive_ratio = torch.div(total_label_entries, num_labels_pos)
    # positive_ratio = torch.div(num_labels_pos, total_label_entries) # pos ratio
    # positive_ratio = torch.reciprocal(positive_ratio)

    #print(num_labels_pos, total_label_entries)
    #negative_ratio = torch.div(total_label_entries-num_labels_pos, total_label_entries)
    #print('postive ratio', negative_ratio, positive_ratio)
    positive_label_impact = torch.mul(positive_ratio,  torch.ones(label_size[0], label_size[1], label_size[2], label_size[3]).cuda())
    #weight_11 = torch.mul(weight_1,  torch.ones(batch_size[0], batch_size[1], batch_size[2]).cuda())
    # binary cross entropy, weight indicates that the less the positive label entries, the more impact the difference between the prediction and the label can have
    criterion = torch.nn.BCELoss(weight = positive_label_impact)#weight = torch.Tensor([0,1]) .cuda() #torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    #loss = class_balanced_cross_entropy_loss(pred, label).cuda()
        
    return criterion(pred, label)

def calc_loss_L1(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    # Variable(label.long()).cuda()
    criterion = torch.nn.L1Loss()#.cuda() #torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    
    return criterion(pred, label)



def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for 
    the last classification layer. Note that for each batchnorm layer, 
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
    any batchnorm parameter
    """
    b = []
    mod = model
    if torch.cuda.device_count() > 1:
        mod = mod.module
    
    if args.use_original_model:
        b.append(mod.encoder.conv1)
        b.append(mod.encoder.bn1)
        b.append(mod.encoder.layer1)
        b.append(mod.encoder.layer2)
        b.append(mod.encoder.layer3)
        b.append(mod.encoder.layer4)
        b.append(mod.encoder.layer5)
        b.append(mod.encoder.main_classifier)
    else:
        b.append(mod.encoder.backbone.conv1)
        b.append(mod.encoder.backbone.bn1)
        b.append(mod.encoder.backbone.layer1)
        b.append(mod.encoder.backbone.layer2)
        b.append(mod.encoder.backbone.layer3)
        b.append(mod.encoder.backbone.layer4)
        b.append(mod.encoder.aspp)
        b.append(mod.encoder.main_classifier)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    mod = model
    if torch.cuda.device_count() > 1:
        mod = model.module

    if args.use_original_model:
        b.append(mod.linear_e.parameters())
        b.append(mod.conv1.parameters())
        b.append(mod.conv2.parameters())
        b.append(mod.gate.parameters())
        b.append(mod.bn1.parameters())
        b.append(mod.bn2.parameters())   
        b.append(mod.main_classifier1.parameters())
        b.append(mod.main_classifier2.parameters())
    else:
        b.append(mod.linear_e.parameters())
        b.append(mod.conv1.parameters())
        b.append(mod.conv2.parameters())
        b.append(mod.gate.parameters())
        b.append(mod.bn1.parameters())
        b.append(mod.bn2.parameters())   
        b.append(mod.main_classifier1.parameters())
        b.append(mod.main_classifier2.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i
            
def lr_poly(base_lr, iter, max_iter, power, epoch):
    if epoch<=2:
        factor = 1
    elif epoch>2 and epoch< 6:
        factor = 1
    else:
        factor = 0.5
    return base_lr*factor*((1-float(iter)/max_iter)**(power))


def netParams(model):
    '''
    Computing total network parameters
    Args:
       model: model
    return: total network parameters
    '''
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        #print(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters


def convert_parameters_for_model(model, saved_state_dict, use_original_model):
    new_params = model.state_dict().copy()
    if args.cuda:
        #model.to(device)
        if use_original_model:
            for i in saved_state_dict["model"]:
                newKey = i.replace("module.", "encoder.")
                new_params[newKey] = saved_state_dict["model"][i]
        else:
            for i in saved_state_dict["model"]:
                if i.startswith("module.layer5."):
                    newKey = i.replace("module.layer5.", "encoder.aspp.")
                elif i.startswith("module.main_classifier."):
                    newKey = i.replace("module.main_classifier.", "encoder.main_classifier.")
                else:
                    newKey = i.replace("module.", "encoder.backbone.")
                new_params[newKey] = saved_state_dict["model"][i]
    return new_params


def main():
    
    print("=====> Configure dataset and pretrained model:",args)
    configure_dataset_init_model(args)
    print(args)

    print("    current dataset:  ", args.dataset)
    print("    init model: ", args.restore_from)
    print("=====> Set GPU for training")
    if args.cuda:
        print("====> Use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    # Select which GPU, -1 if CPU
    #gpu_id = args.gpus
    #device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
    print("=====> Random Seed: ", args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed) 

    if args.output_HW:
        h, w = map(int, args.output_HW.split(','))
        args.output_HW = (h, w)

    w, h = map(int, args.input_size.split(','))
    args.input_size = (w, h)

    cudnn.enabled = True


    print("=====> Loading state saved")

    saved_state_dict = torch.load(args.restore_from)

    print("=====> Building network")

    if args.model == "ori":
        model = CoattentionNet(num_classes=args.num_classes)
        args.use_original_model = True
    else:
        args.use_original_model = False
        model = CoattentionSiameseNet(Bottleneck,3, [3, 4, 23, 3], num_classes=args.num_classes-1)
    #print(model)
    print("=====> Restoring initial state")
    new_params = convert_parameters_for_model(model, saved_state_dict, args.use_original_model)
    # if args.cuda:
    #     #model.to(device)
    #     if torch.cuda.device_count()>1:
    #         for i in saved_state_dict["model"]:
    #             #Scale.layer5.conv2d_list.3.weight
    #             i_parts = i.split('.') # 针对多GPU的情况
    #             #i_parts.pop(1)
    #             #print('i_parts:  ', '.'.join(i_parts[1:-1]))
    #             #if  not i_parts[1]=='main_classifier': #and not '.'.join(i_parts[1:-1]) == 'layer5.bottleneck' and not '.'.join(i_parts[1:-1]) == 'layer5.bn':  #init model pretrained on COCO, class name=21, layer5 is ASPP
                
    #             if i_parts[1].startswith('layer5'):
    #                 key = 'encoder.aspp.' + '.'.join(i_parts[2:])
    #             elif i_parts[1].startswith('main_classifier'):
    #                 key = 'encoder.' + '.'.join(i_parts[1:])
    #             else:
    #                 key = 'encoder.backbone.' + '.'.join(i_parts[1:])
    #             new_params[key] = saved_state_dict["model"][i]
    #         #print('copy {}'.format('.'.join(i_parts[1:])))
    #     else:
    #         for i in saved_state_dict["model"]:
    #             i_parts = i.split('.')
    #             key = 'encoder.' + '.'.join(i_parts[1:])
    #             new_params[key] = saved_state_dict["model"][i]


   
    print("=====> Loading init weights,  pretrained COCO for VOC2012, and pretrained Coarse cityscapes for cityscapes")
 
            
    model.load_state_dict(new_params) #只用到resnet的第5个卷积层的参数


    #print(model.keys())
    if args.cuda:
        #model.to(device)
        if torch.cuda.device_count()>1:
            print("torch.cuda.device_count()=",torch.cuda.device_count())
            model = torch.nn.DataParallel(model).cuda()  #multi-card data parallel
        else:
            print("single GPU for training")
            model = model.cuda()  #1-card data parallel
    start_epoch=0
    
    print("=====> Whether resuming from a checkpoint, for continuing training")
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint["epoch"] 
            model.load_state_dict(checkpoint["model"])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    model.train()
    cudnn.benchmark = True
    
    print('=====> Computing network parameters')
    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))
 
    print("=====> Preparing training data")
    db_train = None
    if args.dataset == 'voc12':
        db_train = VOCDataSet(args.data_dir, args.data_list, max_iters=None, crop_size=args.output_HW, 
                                                 scale=args.random_scale, mirror=args.random_mirror, mean=args.img_mean)
        trainloader = data.DataLoader(db_train, 
                                      batch_size= args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    elif args.dataset == 'cityscapes':
        db_train = CityscapesDataSet(args.data_dir, args.data_list, max_iters=None, crop_size=args.output_HW, 
                                                 scale=args.random_scale, mirror=args.random_mirror, mean=args.img_mean)
        trainloader = data.DataLoader(db_train, 
                                      batch_size = args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    elif args.dataset == 'davis':  #for davis 2016
        db_train = db.PairwiseImg(user_config["train"]["dataset"]["davis"], user_config["train"]["saliency_dataset"], train=True, desired_HW=args.output_HW, db_root_dir=args.data_dir, img_root_dir=args.img_dir,  transform=None, sample_range=args.sample_range, batch_size=args.batch_size) #db_root_dir() --> '/path/to/DAVIS-2016' train path
        trainloader = data.DataLoader(db_train, batch_size= args.batch_size, shuffle=True, num_workers=0)
    else:
        print("dataset error")

    optimizer = optim.SGD([{'params': get_1x_lr_params(model), 'lr': 1*args.learning_rate },  #针对特定层进行学习，有些层不学习
                {'params': get_10x_lr_params(model), 'lr': 10*args.learning_rate}], 
                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()


    args.snapshot_dir = osp.join(".", "snapshots", args.dataset, args.model+"_old_train", 'H'+str(args.output_HW[0])+'W'+str(args.output_HW[1]), ymd_hms)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    
    logFileLoc = os.path.join(args.snapshot_dir,  args.dataset+"_"+args.model+"_old_"+ymd_hms+"_train_log.txt")
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_paramters)))
        logger.write("\n%s\t\t%s" % ('iter', 'Loss(train)\n'))
    logger.flush()

    print("=====> Begin to train")
    train_len=len(trainloader)
    print("  iteration numbers  of per epoch: ", train_len)
    print("  epoch num: ", args.maxEpoches)
    print("  max iteration: ", args.maxEpoches*train_len)
    
    loss_history = []

    for epoch in range(start_epoch, int(args.maxEpoches)):
        
        np.random.seed(args.random_seed + epoch)
        if db_train.train_from_saliency:
            db_train.train_from_saliency(True)
        for i_iter, batch in enumerate(trainloader,0): #i_iter from 0 to len-1
            #print("i_iter=", i_iter, "epoch=", epoch)
            targets, targets_gts, searches, searches_gts = batch['target'], batch['target_gt'], batch['search'], batch['search_gt']
            saliency_images, saliency_gts = batch['img'], batch['img_gt']
            #print(labels.size())
            
            #print(images.size())
            if i_iter%3 ==0: # training from salient images
                # only "img", "img_gt" are required

                #saliency_images.requires_grad_()
                saliency_images = Variable(saliency_images).cuda()
                saliency_gts = Variable(saliency_gts.float().unsqueeze(1)).cuda()
                
                optimizer.zero_grad()

                lr = adjust_learning_rate(optimizer, i_iter+epoch*train_len, epoch,
                    max_iter = args.maxEpoches * train_len)

                pred1, pred2, pred3 = model(saliency_images, saliency_images)
                loss = 0.1*(calc_loss_BCE(pred3, saliency_gts) + 0.8* calc_loss_L1(pred3, saliency_gts) )
                loss.backward()
                if db_train.train_from_saliency:
                    db_train.train_from_saliency(False)
            else:
                # training from video frames
                # only "target", "target_gt", "search", "search_gt" are required
                
                #targets.requires_grad_()
                targets = Variable(targets).cuda()
                targets_gts = Variable(targets_gts.float().unsqueeze(1)).cuda()
                
                #searches.requires_grad_()
                searches = Variable(searches).cuda()
                searches_gts = Variable(searches_gts.float().unsqueeze(1)).cuda()
                
                optimizer.zero_grad()

                lr = adjust_learning_rate(optimizer, i_iter+epoch*train_len, epoch,
                    max_iter = args.maxEpoches * train_len)

                pred1, pred2, pred3 = model(targets, searches)
                loss = calc_loss_BCE(pred1, targets_gts) + 0.8* calc_loss_L1(pred1, targets_gts) + calc_loss_BCE(pred2, searches_gts) + 0.8* calc_loss_L1(pred2, searches_gts)#class_balanced_cross_entropy_loss(pred, labels, size_average=False)
                loss.backward()
                if (i_iter+1)%3 ==0 and db_train.train_from_saliency:
                    db_train.train_from_saliency(True)
            
            optimizer.step()
            
            loss_history.append(loss.data)

            print("===> Epoch[{}]({}/{}): Loss: {:.10f}  lr: {:.5f}".format(epoch, i_iter, train_len, loss.data, lr))
            logger.write("Epoch[{}]({}/{}):     Loss: {:.10f}      lr: {:.5f}\n".format(epoch, i_iter, train_len, loss.data, lr))
            logger.flush()

            del targets
            del targets_gts
            # del target_depth
            del searches
            del searches_gts
            # del search_depth
            del saliency_images
            del saliency_gts
            del batch
            gc.collect()
            torch.cuda.empty_cache()
                
        print("=====> saving model")
        state={"epoch": epoch+1, "model": model.state_dict()}
        torch.save(state, osp.join(args.snapshot_dir, 'co_attention_'+str(args.dataset)+"_"+str(epoch)+'.pth'))


    end = timeit.default_timer()
    print( float(end-start)/3600, 'h')
    logger.write("total training time: {:.2f} h\n".format(float(end-start)/3600))
    logger.close()

    plot2d(np.arange(args.maxEpoches), loss_history, "epoch", "loss", "training_loss_"+args.dataset)


if __name__ == '__main__':
    main()
