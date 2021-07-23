
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
from dataloaders import hzfu_rgbd_loader as rgbddb
import matplotlib.pyplot as plt
import random
import timeit
#from psp.model1 import CoattentionNet  #based on pspnet
# from deeplab.siamese_model_conf import CoattentionNet #siame_model 
#from deeplab.utils import get_1x_lr_params, get_10x_lr_params#, adjust_learning_rate #, loss_calc
from deeplab.residual_net import Bottleneck
# from deeplab.siamese_model import CoattentionSiameseNet
from rgbd_segmentation_model import RGBDSegmentationModel

start = timeit.default_timer()




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
    parser.add_argument("--gpus", type=str, default="3", help="choose gpu device.") 


    return parser.parse_args()

args = get_arguments()


def configure_dataset_init_model(args):
    if args.dataset == 'davis': 
        args.batch_size = user_config["train"]["dataset"]["davis"]["batch_size"]# 1 card: 5, 2 cards: 10 Number of images sent to the network in one step, 16 on paper
        args.maxEpoches = user_config["train"]["dataset"]["davis"]["max_epoches"] # 1 card: 15, 2 cards: 15 epoches, equal to 30k iterations, max iterations= maxEpoches*len(train_aug)/batch_size_per_gpu'),
        args.data_dir = user_config["train"]["dataset"]["davis"]["data_path"]   # 37572 image pairs
        args.img_dir = user_config["train"]["saliency_dataset"] #'/vol/graphics-solar/fengwenb/vos/saliency_dataset'
        args.data_list = './dataset/list/VOC2012/train_aug.txt'  # Path to the file listing the images in the dataset
        args.ignore_label = user_config["train"]["dataset"]["davis"]["ignore_label"]     #The index of the label to ignore during the training
        args.input_size = user_config["train"]["dataset"]["davis"]["input_size"] #'854,480' #Comma-separated string with height and width of images
        args.num_classes = user_config["train"]["dataset"]["davis"]["num_classes"]      #Number of classes to predict (including background)
        args.img_mean = np.array(user_config["train"]["dataset"]["davis"]["img_mean"], dtype=np.float32)       # saving model file and log record during the process of training
        #Where restore model pretrained on other dataset, such as COCO.")
        pretrained_model_name = user_config["train"]["dataset"]["davis"]["pretrained_model"]
        args.restore_from = user_config["train"]["pretrained_models"][pretrained_model_name]["file"]
        #args.restore_from = './pretrained/deep_labv3/deeplab_davis_12_0.pth' #resnet50-19c8e357.pth''/home/xiankai/PSPNet_PyTorch/snapshots/davis/psp_davis_0.pth' #
        args.snapshot_dir = user_config["train"]["dataset"]["davis"]["snapshot_output_path"] #'./snapshots/davis_iteration_conf/'          #Where to save snapshots of the model
        args.resume = user_config["train"]["dataset"]["davis"]["checkpoint_file"] #'./snapshots/davis/co_attention_davis_124.pth' #checkpoint log file, helping recovering training
		
    elif args.dataset == 'hzfurgbd':
        args.batch_size = user_config["train"]["dataset"]["hzfurgbd"]["batch_size"]# 1 card: 5, 2 cards: 10 Number of images sent to the network in one step, 16 on paper
        args.maxEpoches = user_config["train"]["dataset"]["hzfurgbd"]["max_epoches"] # 1 card: 15, 2 cards: 15 epoches, equal to 30k iterations, max iterations= maxEpoches*len(train_aug)/batch_size_per_gpu'),
        args.data_dir = user_config["train"]["dataset"]["hzfurgbd"]["data_path"]   # 37572 image pairs
        args.ignore_label = user_config["train"]["dataset"]["hzfurgbd"]["ignore_label"]     #The index of the label to ignore during the training
        args.input_size = user_config["train"]["dataset"]["hzfurgbd"]["input_size"] #'854,480' #Comma-separated string with height and width of images
        args.num_classes = user_config["train"]["dataset"]["hzfurgbd"]["num_classes"]      #Number of classes to predict (including background)
        args.img_mean = np.array(user_config["train"]["dataset"]["hzfurgbd"]["img_mean"], dtype=np.float32)       # saving model file and log record during the process of training
        #Where restore model pretrained on other dataset, such as COCO.")
        pretrained_model_name = user_config["train"]["dataset"]["hzfurgbd"]["pretrained_model"]
        args.restore_from = user_config["train"]["pretrained_models"][pretrained_model_name]["file"]
        #args.restore_from = './pretrained/deep_labv3/deeplab_davis_12_0.pth' #resnet50-19c8e357.pth''/home/xiankai/PSPNet_PyTorch/snapshots/hzfurgbd/psp_davis_0.pth' #
        args.snapshot_dir = user_config["train"]["dataset"]["hzfurgbd"]["snapshot_output_path"] #'./snapshots/davis_iteration_conf/'          #Where to save snapshots of the model
        args.resume = user_config["train"]["dataset"]["hzfurgbd"]["checkpoint_file"] #'./snapshots/hzfurgbd/co_attention_davis_124.pth' #checkpoint log file, helping recovering training
       
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
    if torch.cuda.device_count() == 1:
        b.append(model.encoder.aspp)
    else:
        b.append(model.module.encoder.backbone.conv1)
        b.append(model.module.encoder.backbone.bn1)
        b.append(model.module.encoder.backbone.layer1)
        b.append(model.module.encoder.backbone.layer2)
        b.append(model.module.encoder.backbone.layer3)
        b.append(model.module.encoder.backbone.layer4)
        b.append(model.module.encoder.aspp)
        b.append(model.module.encoder.main_classifier)
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
    if torch.cuda.device_count() == 1:
        b.append(model.linear_e.parameters())
        b.append(model.main_classifier.parameters())
    else:
        b.append(model.module.depth_encoder.backbone.conv1.parameters())
        b.append(model.module.depth_encoder.backbone.bn1.parameters())
        b.append(model.module.depth_encoder.backbone.layer1.parameters())
        b.append(model.module.depth_encoder.backbone.layer2.parameters())
        b.append(model.module.depth_encoder.backbone.layer3.parameters())
        b.append(model.module.depth_encoder.backbone.layer4.parameters())
        b.append(model.module.depth_encoder.aspp.parameters())
        b.append(model.module.depth_encoder.main_classifier.parameters())

        b.append(model.module.linear_e.parameters())
        b.append(model.module.conv1.parameters())
        b.append(model.module.conv2.parameters())
        b.append(model.module.gate.parameters())
        b.append(model.module.bn1.parameters())
        b.append(model.module.bn2.parameters())   
        b.append(model.module.main_classifier1.parameters())
        b.append(model.module.main_classifier2.parameters())
        
    for j in range(len(b)):
        # print("****b[",j,"]: ",b[j])
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

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True

    print("=====> Loading state saved")

    saved_state_dict = torch.load(args.restore_from)

    print("=====> Building network")

    model = RGBDSegmentationModel(Bottleneck, [3, 4, 23, 3], num_classes=args.num_classes-1)
    # model = CoattentionSiameseNet(Bottleneck,3, [3, 4, 23, 3], num_classes=args.num_classes-1)
    #model = CoattentionNet(num_classes=args.num_classes)
    #print(model)
    new_params = model.state_dict().copy()
    print("=====> Restoring initial state")

    for i in saved_state_dict["model"]:
        #Scale.layer5.conv2d_list.3.weight
        i_parts = i.split('.') # multiple cpu
        #i_parts.pop(1)
        #print('i_parts:  ', '.'.join(i_parts[1:-1]))
        #if  not i_parts[1]=='main_classifier': #and not '.'.join(i_parts[1:-1]) == 'layer5.bottleneck' and not '.'.join(i_parts[1:-1]) == 'layer5.bn':  #init model pretrained on COCO, class name=21, layer5 is ASPP
        
        if i_parts[1].startswith('layer5'):
            key = 'encoder.aspp.' + '.'.join(i_parts[2:])
        elif i_parts[1].startswith('main_classifier'):
            key = 'encoder.' + '.'.join(i_parts[1:])
        else:
            key = 'encoder.backbone.' + '.'.join(i_parts[1:])
        new_params[key] = saved_state_dict["model"][i]
            #print('copy {}'.format('.'.join(i_parts[1:])))
    
   
    print("=====> Loading init weights,  pretrained COCO for VOC2012, and pretrained Coarse cityscapes for cityscapes")
 
            
    model.load_state_dict(new_params) #only resnet first 5 conv layers params
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

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    
    print('=====> Computing network parameters')
    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))
 
    print("=====> Preparing training data")
    db_train = None
    if args.dataset == 'hzfurgbd':
        db_train = rgbddb.HzFuRGBDVideos(user_config["train"]["dataset"]["hzfurgbd"]["data_path"], sample_range=1, desired_HW=input_size, transform=None)
        db_train.set_for_train()
        trainloader = data.DataLoader(db_train, batch_size= args.batch_size, shuffle=True, num_workers=0)
    else:
        print("dataset error")

    optimizer = optim.SGD([{'params': get_1x_lr_params(model), 'lr': 1*args.learning_rate }, 
                {'params': get_10x_lr_params(model), 'lr': 10*args.learning_rate}], 
                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()


    
    logFileLoc = args.snapshot_dir + args.logFile
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
    
    for epoch in range(start_epoch, int(args.maxEpoches)):
        print("......epoch=", epoch)
        db_train.next_batch()
        np.random.seed(args.random_seed + epoch)
        for i_iter, batch in enumerate(trainloader,0): #i_iter from 0 to len-1
            db_train.next_batch()
            print("  i_iter=", i_iter)
            current_rgb, current_depth, current_gt, counterpart_rgb, counterpart_depth, counterpart_gt = batch['target'], batch['target_depth'], batch['target_gt'], batch['search_0'], batch['search_0_depth'], batch['search_0_gt'],

            
            current_rgb.requires_grad_()
            current_rgb = Variable(current_rgb).cuda()
            current_depth.requires_grad_()
            current_depth = Variable(current_depth).cuda()
            current_gt = Variable(current_gt.float().unsqueeze(1)).cuda()

            counterpart_rgb.requires_grad_()
            counterpart_rgb = Variable(counterpart_rgb).cuda()
            counterpart_depth.requires_grad_()
            counterpart_depth = Variable(counterpart_depth).cuda()
            counterpart_gt = Variable(counterpart_gt.float().unsqueeze(1)).cuda()

            optimizer.zero_grad()
            
            lr = adjust_learning_rate(optimizer, i_iter+epoch*train_len, epoch,
                    max_iter = args.maxEpoches * train_len)
            #print(images.size())

            pred1, pred2, pred3 = model(current_rgb, counterpart_rgb, current_depth, counterpart_depth)
            loss = calc_loss_BCE(pred1, current_gt) + 0.8* calc_loss_L1(pred1, current_gt) + calc_loss_BCE(pred2, counterpart_gt) + 0.8* calc_loss_L1(pred2, counterpart_gt)#class_balanced_cross_entropy_loss(pred, labels, size_average=False)
            loss.backward()
            
            optimizer.step()
                
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}  lr: {:.5f}".format(epoch, i_iter, train_len, loss.data, lr))
            logger.write("Epoch[{}]({}/{}):     Loss: {:.10f}      lr: {:.5f}\n".format(epoch, i_iter, train_len, loss.data, lr))
            logger.flush()
                
        print("=====> saving model")
        state={"epoch": epoch+1, "model": model.state_dict()}
        torch.save(state, osp.join(args.snapshot_dir, 'co_attention_rgbd_'+str(args.dataset)+"_"+str(epoch)+'.pth'))


    end = timeit.default_timer()
    print( float(end-start)/3600, 'h')
    logger.write("total training time: {:.2f} h\n".format(float(end-start)/3600))
    logger.close()


if __name__ == '__main__':
    main()
