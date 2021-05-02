# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 11:39:54 2018

@author: carri
"""

from __future__ import division

import os
import numpy as np
import cv2
from scipy.misc import imresize
import scipy.misc 
import random

#from dataloaders.helpers import *
from torch.utils.data import Dataset

def flip(I,flip_p):
    if flip_p>0.5:
        return np.fliplr(I)
    else:
        return I

def scale_im(img_temp,scale):
    new_dims = (  int(img_temp.shape[0]*scale),  int(img_temp.shape[1]*scale)   )
    return cv2.resize(img_temp,new_dims).astype(float)

def scale_gt(img_temp,scale):
    new_dims = (  int(img_temp.shape[0]*scale),  int(img_temp.shape[1]*scale)   )
    return cv2.resize(img_temp,new_dims,interpolation = cv2.INTER_NEAREST).astype(float)

def my_crop(img,gt):
    H = int(0.9 * img.shape[0])
    W = int(0.9 * img.shape[1])
    H_offset = random.choice(range(img.shape[0] - H))
    W_offset = random.choice(range(img.shape[1] - W))
    H_slice = slice(H_offset, H_offset + H)
    W_slice = slice(W_offset, W_offset + W)
    img = img[H_slice, W_slice, :]
    gt = gt[H_slice, W_slice]
    
    return img, gt

class PairwiseImg(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, dataset_config, saliency_dataset_config, train=True,
                 inputRes=None,
                 db_root_dir='/DAVIS-2016',
                 img_root_dir = None,
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892),
                 seq_name=None, sample_range=10):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """
        self.train = train
        self.range = sample_range
        self.inputRes = inputRes
        self.img_root_dir = img_root_dir
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.meanval = meanval
        self.seq_name = seq_name

        self.dataset_config = dataset_config
        self.saliency_dataset_config = saliency_dataset_config

        if self.train:
            fname = 'train_seqs'
        else:
            fname = 'val_seqs'

        if self.seq_name is None: #所有的数据集都参与训练
            with open(dataset_config["subset_file"]) as f:
	    #with open(os.path.join(db_root_dir, fname + '.txt')) as f: #wbf
                seqs = f.readlines()
                video_list = []
                labels = []
                Index = {}
                image_list = [] #DUTS-TR/Imgs/0001.jpg
                im_label = []
                for seq in seqs:
                    print("seq: ",seq) #wbf
                    path_to_category = os.path.join(dataset_config["img_path"], seq.strip('\n'))
                    images = np.sort(os.listdir(path_to_category))
                    images_path = list(map(lambda x: os.path.join(seq.strip(), x), images))
                    print("images_path",images_path) #wbf
                    start_num = len(video_list)
                    video_list.extend(images_path)
                    end_num = len(video_list)
                    Index[seq.strip('\n')]= np.array([start_num, end_num])
                    lab = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', seq.strip('\n'))))
                    lab_path = list(map(lambda x: os.path.join('Annotations/480p/', seq.strip(), x), lab))
                    labels.extend(lab_path)
                    
                saliency_datasets = dataset_config["saliency_datasets"]
                #data_list = np.sort(os.listdir(db_root_dir))
                for seq in saliency_datasets: #所有数据集
                    seq = seq.strip('\n') 
                    print(" saliency seq: ",seq) #wbf
                    img_fullpath = os.path.join(saliency_dataset_config["root_path"], saliency_dataset_config["datasets"][seq]["images"])
                    images = np.sort(os.listdir(img_fullpath))#针对某个数据集，比如DUT			
        # Initialize the original DAVIS splits for training the parent network
                    images_path = list(map(lambda x: os.path.join(saliency_dataset_config["datasets"][seq]["images"], x), images))         
                    image_list.extend(images_path)

                    lab_fullpath = os.path.join(saliency_dataset_config["root_path"], saliency_dataset_config["datasets"][seq]["masks"])
                    lab = np.sort(os.listdir(lab_fullpath))
                    lab_path = list(map(lambda x: os.path.join(saliency_dataset_config["datasets"][seq]["masks"],x), lab))
                    im_label.extend(lab_path)

        else: #针对所有的训练样本， video_list存放的是图片的路径

            # Initialize the per sequence images for online training
            names_img = np.sort(os.listdir(os.path.join(db_root_dir, str(seq_name))))
            video_list = list(map(lambda x: os.path.join(( str(seq_name)), x), names_img))
            #name_label = np.sort(os.listdir(os.path.join(db_root_dir,  str(seq_name))))
            labels = [os.path.join( (str(seq_name)+'/saliencymaps'), names_img[0])]
            labels.extend([None]*(len(names_img)-1)) #在labels这个列表后面添加元素None
            if self.train:
                video_list = [video_list[0]]
                labels = [labels[0]]

        assert (len(labels) == len(video_list))

        self.video_list = video_list
        self.labels = labels
        self.saliency_images = image_list
        self.saliency_labels = im_label
        self.Index = Index
        #img_files = open('all_im.txt','w+')

    def __len__(self):
        print(len(self.video_list), len(self.saliency_images))
        return len(self.video_list)
    
    def __getitem__(self, idx):
        target, target_gt = self.make_video_gt_pair(idx)
        target_id = idx
        img_idx = np.random.randint(1,len(self.saliency_images)-1)
        seq_name1 = self.video_list[idx].split('/')[0] #获取物体名称
        if self.train:
            my_index = self.Index[seq_name1]
            search_id = np.random.randint(my_index[0], my_index[1])#min(len(self.video_list)-1, target_id+np.random.randint(1,self.range+1))
            if search_id == target_id:
                search_id = np.random.randint(my_index[0], my_index[1])
            search, search_gt = self.make_video_gt_pair(search_id)
            img, img_gt = self.make_img_gt_pair(img_idx)
            sample = {'target': target, 'target_gt': target_gt, 'search': search, 'search_gt': search_gt, \
                      'img': img, 'img_gt': img_gt}
            #np.save('search1.npy',search)
            #np.save('search_gt.npy',search_gt)
            if self.seq_name is not None:
                fname = os.path.join(self.seq_name, "%05d" % idx)
                sample['fname'] = fname

            if self.transform is not None:
                sample = self.transform(sample)
       
        else:
            img, gt = self.make_video_gt_pair(idx)
            sample = {'image': img, 'gt': gt}
            if self.seq_name is not None:
                fname = os.path.join(self.seq_name, "%05d" % idx)
                sample['fname'] = fname
        
        
        
        return sample  #这个类最后的输出

    def make_video_gt_pair(self, idx): #这个函数存在的意义是为了getitem函数服务的
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread(os.path.join(self.dataset_config["img_path"], self.video_list[idx]), cv2.IMREAD_COLOR)
        if self.labels[idx] is not None and self.train:
            label = cv2.imread(os.path.join(self.db_root_dir, self.labels[idx]), cv2.IMREAD_GRAYSCALE)
            #print(os.path.join(self.db_root_dir, self.labels[idx]))
        else:
            gt = np.zeros(img.shape[:-1], dtype=np.uint8)
            
         ## 已经读取了image以及对应的ground truth可以进行data augmentation了
        if self.train:  #scaling, cropping and flipping
             img, label = my_crop(img,label)
             scale = random.uniform(0.7, 1.3)
             flip_p = random.uniform(0, 1)
             img_temp = scale_im(img,scale)
             img_temp = flip(img_temp,flip_p)
             gt_temp = scale_gt(label,scale)
             gt_temp = flip(gt_temp,flip_p)
             
             img = img_temp
             label = gt_temp
             
        if self.inputRes is not None:
            img = imresize(img, self.inputRes)
            #print('ok1')
            #scipy.misc.imsave('label.png',label)
            #scipy.misc.imsave('img.png',img)
            if self.labels[idx] is not None and self.train:
                label = imresize(label, self.inputRes, interp='nearest')

        img = np.array(img, dtype=np.float32)
        #img = img[:, :, ::-1]
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))        
        img = img.transpose((2, 0, 1))  # NHWC -> NCHW
        
        if self.labels[idx] is not None and self.train:
                gt = np.array(label, dtype=np.int32)
                gt[gt!=0]=1
                #gt = gt/np.max([gt.max(), 1e-8])
        #np.save('gt.npy')
        return img, gt

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.dataset_config["img_path"], self.video_list[0]))
        
        return list(img.shape[:2])

    def make_img_gt_pair(self, idx): #这个函数存在的意义是为了getitem函数服务的
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread(os.path.join(self.saliency_dataset_config["root_path"], self.saliency_images[idx]),cv2.IMREAD_COLOR)
        #print(os.path.join(self.db_root_dir, self.img_list[idx]))
        if self.img_labels[idx] is not None and self.train:
            label = cv2.imread(os.path.join(self.img_root_dir, self.saliency_labels[idx]),cv2.IMREAD_GRAYSCALE)
            #print(os.path.join(self.db_root_dir, self.labels[idx]))
        else:
            gt = np.zeros(img.shape[:-1], dtype=np.uint8)
            
        if self.inputRes is not None:            
            img = imresize(img, self.inputRes)
            if self.img_labels[idx] is not None and self.train:
                label = imresize(label, self.inputRes, interp='nearest')

        img = np.array(img, dtype=np.float32)
        #img = img[:, :, ::-1]
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))        
        img = img.transpose((2, 0, 1))  # NHWC -> NCHW
        
        if self.img_labels[idx] is not None and self.train:
                gt = np.array(label, dtype=np.int32)
                gt[gt!=0]=1
                #gt = gt/np.max([gt.max(), 1e-8])
        #np.save('gt.npy')
        return img, gt
    
if __name__ == '__main__':
    import custom_transforms as tr
    import torch
    from torchvision import transforms
    from matplotlib import pyplot as plt

    transforms = transforms.Compose([tr.RandomHorizontalFlip(), tr.Resize(scales=[0.5, 0.8, 1]), tr.ToTensor()])

    #dataset = DAVIS2016(db_root_dir='/media/eec/external/Databases/Segmentation/DAVIS-2016',
                       # train=True, transform=transforms)
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
#
#    for i, data in enumerate(dataloader):
#        plt.figure()
#        plt.imshow(overlay_mask(im_normalize(tens2image(data['image'])), tens2image(data['gt'])))
#        if i == 10:
#            break
#
#    plt.show(block=True)
