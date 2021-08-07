# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 11:39:54 2018

@author: carri
"""
# for testing case
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

    def __init__(self, train=True,
                 output_HW=None,
                 db_root_dir='/DAVIS-2016',
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892),
                 seq_name=None, sample_range=10):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """

        self.paths_of_images = None
        self.paths_of_labels = None
        self.offsets_of_seqs = None

        self.train = train
        self.range = sample_range
        self.output_HW = output_HW # w, h
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.meanval = meanval
        self.seq_name = seq_name

        if self.train:
            fname = 'train'
        else:
            fname = 'val'

	    print("seq name: ",self.seq_name, " root: ",db_root_dir, " fname: ",fname)

        if self.seq_name is None: #所有的数据集都参与训练
            with open(os.path.join(db_root_dir, 'ImageSets', '480p', fname + '.txt')) as f:
                seqs = f.readlines()
                images_paths = []
                labels_paths = []
                offsets_of_seqs = {}
                for seq in seqs:
                    #print("crt seq: ", seq)
                    parts = seq.strip('\n').split()
                    parts[0] = parts[0][1:] # remove /    RGB path
                    parts[1] = parts[1][1:] # remove /    label path
                    part_img_path = parts[0]
                    subfolder = part_img_path.split('/') # JPEGImages, 480p, dance-twirl, 00073.jpg
                    # print("subfol: ", subfolder)

                    #images = np.sort(os.listdir(os.path.join(db_root_dir, subfolder[1], subfolder[2], subfolder[3]))) #seq.strip('\n'))))
                    #print("images: ",images)
                    images_path = os.path.join(db_root_dir, parts[0]) #subfolder[1], subfolder[2]) #list(map(lambda x: os.path.join(db_root_dir), images))
                    # images_path is db_root_dir + RGB path
                    
                    #print("imag path: ",images_path)
                    # print(" subfold ",subfolder)
                    if subfolder[2] not in offsets_of_seqs:
                        start_num = len(images_paths)
                        images_paths.append(images_path)
                        end_num = len(images_paths)
                        offsets_of_seqs[subfolder[2]]= np.array([start_num, end_num])
                    else:
                        images_paths.append(images_path)
                        end_num = len(images_paths)
                        offsets_of_seqs[subfolder[2]][1] = end_num
                
                    lab_path = os.path.join(db_root_dir, parts[1])	
                    #lab = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', seq.strip('\n'))))
                    #lab_path = list(map(lambda x: os.path.join('Annotations/480p/', seq.strip(), x), lab))
                    # print("label path: ",lab_path)
                    labels_paths.append(lab_path)
                        #print("labels_paths: ",labels_paths)
        else: #针对所有的训练样本， img_list存放的是图片的路径

            # Initialize the per sequence images for online training
            names_img = np.sort(os.listdir(os.path.join(db_root_dir, str(seq_name))))
            images_paths = list(map(lambda x: os.path.join(( str(seq_name)), x), names_img))
            #name_label = np.sort(os.listdir(os.path.join(db_root_dir,  str(seq_name))))
            labels_paths = [os.path.join( (str(seq_name)+'/saliencymaps'), names_img[0])]
            labels_paths.extend([None]*(len(names_img)-1)) #在labels这个列表后面添加元素None
            if self.train:
                images_paths = [images_paths[0]]
                labels_paths = [labels_paths[0]]

        # print(labels_paths)
        # print("======")
        # print(images_paths)

        assert (len(labels_paths) == len(images_paths))
        # print(images_paths)

        self.paths_of_images = images_paths
        self.paths_of_labels = labels_paths
        self.offsets_of_seqs = offsets_of_seqs
        #img_files = open('all_im.txt','w+')

    def __len__(self):
        return len(self.paths_of_images)

    def __getitem__(self, idx):
        target, target_gt,sequence_name = self.make_img_gt_pair(idx, True) #测试时候要分割的帧
        target_id = idx
        seq_name1 = self.paths_of_images[target_id].split('/')[-2] #获取视频名称
        # print("seq name1: ",seq_name1, idx)
        sample = {'target': target, 'target_gt': target_gt, 'seq_name': sequence_name, 'search_0': None, 'frame_index': idx}
        if self.range>=1:
            my_index = self.offsets_of_seqs[seq_name1]
            search_num = list(range(my_index[0], my_index[1]))  
            search_ids = random.sample(search_num, self.range)#min(len(self.paths_of_images)-1, target_id+np.random.randint(1,self.range+1))
        
            for i in range(0,self.range):
                search_id = search_ids[i]
                search, search_gt,sequence_name = self.make_img_gt_pair(search_id, False)
                if sample['search_0'] is None:
                    sample['search_0'] = search
                else:
                    sample['search'+'_'+str(i)] = search
            #np.save('search1.npy',search)
            #np.save('search_gt.npy',search_gt)
            if self.seq_name is not None:
                fname = os.path.join(self.seq_name, "%05d" % idx)
                sample['fname'] = fname
       
        else:
            img, _ = self.make_img_gt_pair(idx, False)
            sample['search_0'] = img
            if self.seq_name is not None:
                fname = os.path.join(self.seq_name, "%05d" % idx)
                sample['fname'] = fname

        return sample  #这个类最后的输出

    def make_img_gt_pair(self, idx, load_gt): #这个函数存在的意义是为了getitem函数服务的
        """
        Make the image-ground-truth pair
        """
        need_gt = load_gt and self.paths_of_labels[idx] is not None #and self.train we need to calc the accuracy.
        img = cv2.imread(os.path.join(self.db_root_dir, self.paths_of_images[idx]), cv2.IMREAD_COLOR)
        if need_gt:
            label = cv2.imread(os.path.join(self.db_root_dir, self.paths_of_labels[idx]), cv2.IMREAD_GRAYSCALE)
            #print(os.path.join(self.db_root_dir, self.paths_of_labels[idx]))
        else:
            gt = np.zeros((1,1), dtype=np.uint8) #((img.shape[:-1]), dtype=np.uint8)
            
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
             
        if self.output_HW is not None:
            img = imresize(img, self.output_HW)
            #print('ok1')
            #scipy.misc.imsave('label.png',label)
            #scipy.misc.imsave('img.png',img)
            if need_gt:
                label = imresize(label, self.output_HW, interp='nearest')

        img = np.array(img, dtype=np.float32)
        #img = img[:, :, ::-1]
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))
        img = img.transpose((2, 0, 1))  # NHWC -> NCHW
        
        if need_gt:
                gt = np.array(label, dtype=np.int32)
                gt[gt!=0]=1
                #gt = gt/np.max([gt.max(), 1e-8])
        #np.save('gt.npy')
        sequence_name = self.paths_of_images[idx].split('/')[-2]
        return img, gt, sequence_name 

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.paths_of_images[0]))
        
        return list(img.shape[:2])


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
