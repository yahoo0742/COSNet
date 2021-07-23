# https://hzfu.github.io/proj_rgbdseg.html
# The dataset for Foreground Segmentation in RGBD Video

# Please cite the following publications if you used this work:

# [1] Huazhu Fu, Dong Xu, Stephen Lin, and Jiang Liu, "Object-based RGBD Image Co-segmentation with Mutex Constraint," in CVPR, 2015, pp. 4428-4436.
# [2] Huazhu Fu, Dong Xu, Stephen Lin, "Object-based Multiple Foreground Segmentation in RGBD Video", Submitted to IEEE Transactions on Image Processing, 2016.

# ====================================================================
# This dataset includes:
# 1) RGB_data: RGB video frames.
# 2) Depth_data: Depth map for video frame with '.mat' format. It contains original depth (depth_org), and smoothed depth map (depth) by using 'NYU Depth V2 Dataset Matlab Toolbox'.
# 3) Label: The groundtruth of video frame.
# 4) our_result: our results in [2].

# |-RGB_data
# | |-seq_1_name
# | | |-01.png
# | | |-02.png
# | | |-...png
# | |-seq_2_name
# | | |-01.png
# | | |-...png
# | |-seq_...
# |
# |-Depth_data
# | |-seq_1_name
# | | |-01.mat
# | | |-02.mat
# | | |-...mat
# | |-seq_2_name
# | | |-01.mat
# | | |-...mat
# | |-seq_...
# |
# |-Label
# | |-seq_1_name
# | | |-01.png
# | | |-02.png
# | |...

import os
import numpy as np
import random
import cv2
import h5py
from scipy.misc import imresize
from torch.utils.data import Dataset
from dataloaders import utils

k_sub_set_percentage = {
    'train': 0.8,
    'test': 0.2
}

class Folder:
    def __init__(self, folder_name):
        self.folder_name = folder_name
        self.extension_name = None

    def set_extension_name(self, extension_name):
        self.extension_name = extension_name

class EContentInfo:
    rgb = Folder('RGB_data') #, 'png')
    depth = Folder('Depth_data') #, 'mat')
    groundtruth = Folder('Label') #, 'png')
    all_folders = [rgb, depth, groundtruth]

class VideoFrameInfo:
    def __init__(self, seq_name, id, name_rgb_frame, name_depth_frame, name_gt_frame):
        self.id = id
        self.name_of_rgb_frame = name_rgb_frame
        self.name_of_depth_frame = name_depth_frame
        self.name_of_groundtruth_frame = name_gt_frame
        self.seq_name = seq_name
    def __str__(self):
        return self.seq_name+"/["+self.id+"]:"+self.name_of_rgb_frame+","+self.name_of_groundtruth_frame

  
class HzFuRGBDVideos(Dataset):
    def __init__(self, dataset_root,
                 sample_range,
                 desired_HW=None,
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892),
                 ):
        self.dataset_root = dataset_root
        self.sample_range = sample_range
        self.desired_HW = desired_HW # H W
        self.transform = transform
        self.meanval = meanval
        self.stage = 'train'
        self.flip_seq_for_augmentation = {} # seq_name: flip_probability. During training, images can be flipped horizontally for augmentation. Frames of a sequence should be all flipped or all not flipped.

        self.sets = {
            'entire': {
                'names_of_sequences': [],
                'offset4_frames_of_sequences': {},
                'names_of_frames': [],
                'frame_to_sequence': []
            },
            'train': {
                'names_of_sequences': [],
                'offset4_frames_of_sequences': {},
                'names_of_frames': [], # it is more convenient for iterating frames by organizing all frames of all sequence together 
                'frame_to_sequence': []
            },
            'validate': {
                'names_of_sequences': [],
                'offset4_frames_of_sequences': {},
                'names_of_frames': [],
                'frame_to_sequence': []
            },
            'test': {
                'names_of_sequences': [],
                'offset4_frames_of_sequences': {},
                'names_of_frames': [],
                'frame_to_sequence': []
            }
        }

        self._load_meta_data()
        self._split_dataset()

    def set_for_test(self):
        self.stage = 'test'
    
    def set_for_train(self):
        self.stage = 'train'

    def new_training_epoch(self):
        self.flip_seq_for_augmentation.clear()

    def _get_path(self, content_type, seq_name=None, frame_name=None): #(self, content_type:Folder, seq_name=None, frame_name=None):
        if seq_name == None:
            return os.path.join(self.dataset_root, content_type.folder_name)
        if seq_name not in self.sets['entire']['names_of_sequences']:
            raise Exception('Cannot find sequence ' + seq_name)
        if frame_name == None:
            return os.path.join(self.dataset_root, content_type.folder_name, seq_name)

        # frames_of_seq = self._get_frames_of_seq(seq_name)
        # if frames_of_seq == None:
        #     raise Exception('Cannot find any frame for ' + seq_name)
        # if frame_name not in frames_of_seq:
        #     raise Exception('Frame ' + frame_name + ' is not present in sequence ' + seq_name)

        return os.path.join(self.dataset_root, content_type.folder_name, seq_name, frame_name)

    def _get_path_of_rgb_data(self, seq_name=None, frame_name=None):
        return self._get_path(EContentInfo.rgb, seq_name, frame_name)

    def _get_path_of_depth_data(self, seq_name=None, frame_name=None):
        return self._get_path(EContentInfo.depth, seq_name, frame_name)

    def _get_path_of_groundtruth_data(self, seq_name=None, frame_name=None):
        return self._get_path(EContentInfo.groundtruth, seq_name, frame_name)

    def _get_frames_of_seq(self, set_name, seq_name):
        seq_offset = self.sets[set_name]['offset4_frames_of_sequences'][seq_name]
        if seq_offset:
            return self.sets[set_name]['names_of_frames'][seq_offset['start']:seq_offset['end']]
        return None

    def _get_framename_by_index(self, set_name, frame_index):
        if frame_index >= len(self.sets[set_name]['names_of_frames']):
            return None
        return self.sets[set_name]['names_of_frames'][frame_index]

    def _load_meta_data(self):
        rgb_data_path = self._get_path_of_rgb_data()
        self.sets['entire']['names_of_sequences'] = os.listdir(rgb_data_path)

        def __remove_extension_name(fullname): #(fullname:str):
            ext_starts = fullname.rindex('.')
            if ext_starts > 0:
                ext = fullname[ext_starts+1:]
                return (fullname[:ext_starts], ext)
            raise Exception(fullname + ' does not have an extension name.')
  
        def __get_extension_name(fullname): #(fullname:str):
            ext_starts = fullname.rindex('.')
            if ext_starts > 0:
                return fullname[ext_starts+1:]
            return None

        def __is_extension_name(fullname, ext_name): #(fullname:str, ext_name:str):
            my_ext = __get_extension_name(fullname)
            return my_ext == ext_name

        def __check_framenames_of_sequence(folder, seq): #(folder:Folder, seq):
            seq_path = self._get_path(folder, seq)
            framenames_of_seq = os.listdir(seq_path)
            if False and len(framenames_of_seq) > 0:
                '''
                check whether all frames have the same extension name, if they have the same extension name, remove the extension name from
                the frame names.
                '''
                test_ext = __get_extension_name(framenames_of_seq[0])
                same_ext_name_for_all = all([__is_extension_name(name, test_ext) for name in framenames_of_seq])
                if same_ext_name_for_all:
                    if folder.extension_name == None or test_ext == folder.extension_name:
                        # all frames have the same extension name, remove the extension name from the frame names
                        framenames_of_seq = [__remove_extension_name(fullname, test_ext) for fullname in framenames_of_seq]
                        # save the extension name
                        folder.set_extension_name(test_ext)
                    else:
                        folder.set_extension_name(None)
                else:
                    folder.set_extension_name(None)
            return framenames_of_seq

        for seq in self.sets['entire']['names_of_sequences']:
            '''
            because not every rgb frame has been labelled in the dataset and I only consider the labelled frames as valid frames for training and test,
            I need to collect rgb frames that have corresponding grountruth frame
            '''
            names_of_rgb_frames_of_seq = __check_framenames_of_sequence(EContentInfo.rgb, seq)
            names_of_depth_frames_of_seq = __check_framenames_of_sequence(EContentInfo.depth, seq)
            names_of_gt_frames_of_seq = __check_framenames_of_sequence(EContentInfo.groundtruth, seq)

            names_of_rgb_frames_of_seq.sort()
            names_of_depth_frames_of_seq.sort()
            # assume every rgb frame has a corresponding unique depth frame
            assert(len(names_of_depth_frames_of_seq) == len(names_of_rgb_frames_of_seq), 'RGB frames of '+seq+" are different from depth frames.")

            labelled_frames_of_seq = []
            ids_collected = []
            # the name of a ground truth frame is like XX_obj_Y.png, XX is the id of the gt frame, and it is also the id of a corresponding frame under RGB_data, Y is the index of the salient object of the frame
            ids_of_gt_frames = [frame[:2] for frame in names_of_gt_frames_of_seq] # We know there are duplicate frames (XX is present more than once)
            ids_of_gt_frames = set(ids_of_gt_frames) # unique, ignore the value of Y, only focus on one salient object

            for gt_frame_id in ids_of_gt_frames:
                if gt_frame_id not in ids_collected: # this is a naive rule of choosing the first salient object of the frame as the target
                    # this labelled frame hasn't been collected, collect it
                    ids_collected.append(gt_frame_id)

                    rgb_framename = None
                    depth_framename = None
                    gt_framename = None

                    # find the full rgb frame name and the full depth frame name that have this id
                    for i in range(len(names_of_rgb_frames_of_seq)):
                        if names_of_rgb_frames_of_seq[i].startswith(gt_frame_id):
                            # found the rgb frame
                            rgb_framename = names_of_rgb_frames_of_seq[i]
                            depth_framename = names_of_depth_frames_of_seq[i]
                            assert(depth_framename.startswith(gt_frame_id), 'RGB frame ' + rgb_framename + ' of '+seq+' does not have a corresponding from depth frame.')
                            names_of_rgb_frames_of_seq.pop(i) # delete the frame that has been collected, so that for matching next frame id, we can speed up
                            names_of_depth_frames_of_seq.pop(i) # delete the frame that has been collected, so that for matching next frame id, we can speed up
                            break

                    # find the full name of a gt frame that has the id
                    for i in range(len(names_of_gt_frames_of_seq)):
                        if names_of_gt_frames_of_seq[i].startswith(gt_frame_id):
                            gt_framename = names_of_gt_frames_of_seq[i]
                            names_of_gt_frames_of_seq.pop(i) # delete the frame that has been collected, so that for matching next frame id, we can speed up
                            break
                    
                    assert(rgb_framename!= None and depth_framename != None and gt_framename != None, 'Cannot find the frame for id '+gt_frame_id)
                    videoFrameInfo = VideoFrameInfo(seq, gt_frame_id, rgb_framename, depth_framename, gt_framename)
                    labelled_frames_of_seq.append(videoFrameInfo)

            # now all valid frames (labelled frames) of this sequence have been collected in labelled_frames_of_seq

            if len(labelled_frames_of_seq) > 0:
                start_idx = len(self.sets['entire']['names_of_frames'])
                end_idx = start_idx + len(labelled_frames_of_seq)
                self.sets['entire']['offset4_frames_of_sequences'][seq] = {'start': start_idx, 'end': end_idx}
                self.sets['entire']['names_of_frames'].extend(labelled_frames_of_seq)

    def _split_dataset(self):
        # sequence based -- all frames of a sequence is randomly chosen for training, or for test
        for seq in self.sets['entire']['names_of_sequences']:
            rand = random.random()
            frames_of_seq = self._get_frames_of_seq('entire', seq)
            if not frames_of_seq:
                raise Exception('Cannot find any frame for '+seq)

            if rand < k_sub_set_percentage['train']:
                # train
                to_be_in_subset = 'train'
            else:
                # test
                to_be_in_subset = 'test'

            seq_index = len(self.sets[to_be_in_subset]['names_of_sequences'])
            self.sets[to_be_in_subset]['names_of_sequences'].append(seq) # add this sequence to this set
  
            start_idx = len(self.sets[to_be_in_subset]['names_of_frames'])
            num_frames = len(frames_of_seq)
            end_idx = num_frames + start_idx
            self.sets[to_be_in_subset]['offset4_frames_of_sequences'][seq] = {'start': start_idx, 'end': end_idx}
            self.sets[to_be_in_subset]['names_of_frames'].extend(frames_of_seq)
            #self.sets[to_be_in_subset]['frame_to_sequence'].extend([seq_index]*num_frames)

    # implementation
    def __len__(self):
        set_name = self.stage
        print("HzFuRGBDVideos length: " , len(self.sets[set_name]['names_of_frames']))
        return len(self.sets[set_name]['names_of_frames'])

    def __getitem__(self, frame_index):
        set_name = self.stage
        frame_info = self._get_framename_by_index(set_name, frame_index)
        if frame_info:
            sample = {'target': None, 'target_depth': None, 'target_gt': None, 'seq_name': None, 'search_0': None, 'search_0_depth': None, 'search_0_gt':None, 'frame_index': frame_info.id}
            current_img, current_depth, current_img_gt = self._load_rgbd_and_gt(frame_info)
            sample['target'] = current_img
            sample['target_depth'] = current_depth
            sample['target_gt'] = current_img_gt
            sample['seq_name'] = frame_info.seq_name

            _range = self.sets[set_name]['offset4_frames_of_sequences'][frame_info.seq_name]

            if self.sample_range > 1:
                search_ids_pool= list(range(_range['start'], _range['end']))
                search_ids = random.sample(search_ids_pool, self.sample_range)#min(len(self.img_list)-1, target_id+np.random.randint(1,self.range+1))

                for i in range(0,self.sample_range):
                    search_id = search_ids[i]
                    frame_info_to_match = self._get_framename_by_index(set_name, search_id)
                    match_img, match_depth, match_img_gt = self._load_rgbd_and_gt(frame_info_to_match)
                    sample['search'+'_'+str(i)] = match_img
                    sample['search'+'_'+str(i)+'_depth'] = match_depth
                    sample['search'+'_'+str(i)+'_gt'] = match_img_gt

            else:
                idx_to_match = frame_index
                if _range['start'] < _range['end'] -1:
                    count = 0
                    while idx_to_match == frame_index:
                        count += 1
                        if count > 3:
                            idx_to_match = frame_index + 1
                            break
                        idx_to_match = random.randint(_range['start'], _range['end']-1)
                if idx_to_match == frame_index:
                    print("Got a pair of frames with the same index "+frame_index+". The frame is about "+frame_info)
                else:
                    frame_info_to_match = self._get_framename_by_index(set_name, idx_to_match)
                    match_img, match_depth, match_img_gt = self._load_rgbd_and_gt(frame_info_to_match)
                    sample['search_0'] = match_img
                    sample['search_0_depth'] = match_depth
                    sample['search_0_gt'] = match_img_gt

            print(" ##### sample rgb: ",sample['target'].shape, " gt: ", sample['target_gt'].shape,  " depth: ", sample['target_depth'].shape, " search_rgb: " ,sample['search_0'].shape, " search_0_gt: ",sample['search_0_gt'].shape,  "search_0_depth: ", sample['search_0_depth'].shape)
            return sample
            # return current_img, current_depth, current_img_gt, match_img, match_depth, match_img_gt

        else:
            raise Exception('Cannot find the sequence from frame index ', frame_index)

    def _load_rgbd_and_gt(self, frame_info):
    #def _load_rgbd_and_gt(self, frame_index_in_train_set):
        def __load_mat(path): #->np.array:
            # in the shape of (H, W) with values in [0, 255]
            f = h5py.File(path, 'r')
            result = np.array(f['depth'], dtype=np.float32)
            print("depth shape: ",result.shape)

            if self.desired_HW is not None:
                result = imresize(result, self.desired_HW)
            print(" after depth shape: ",result.shape)
            result = (result - result.min()) * 255 / (result.max() - result.min())
            # result = result.transpose() 
            return result

        rgb_path = self._get_path_of_rgb_data(frame_info.seq_name, frame_info.name_of_rgb_frame)
        depth_path = self._get_path_of_depth_data(frame_info.seq_name, frame_info.name_of_depth_frame)
        gt_path = self._get_path_of_groundtruth_data(frame_info.seq_name, frame_info.name_of_groundtruth_frame)

        if rgb_path and depth_path and gt_path:
            rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            if self.desired_HW is not None:
                rgb_img = imresize(rgb_img, self.desired_HW)

            rgb_img = np.array(rgb_img, dtype=np.float32)
            rgb_img = np.subtract(rgb_img, np.array(self.meanval, dtype=np.float32)) 
            rgb_img = rgb_img.transpose((2, 0, 1))  # HWC -> CHW

            depth_img = __load_mat(depth_path)
            depth_img = depth_img[None, :,:] # 1, H, W with values in [0, 255]

            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if self.desired_HW is not None:
                gt = imresize(gt, self.desired_HW)
            gt[gt!=0]=1 # H, W with values in {0, 1}
            gt = np.array(gt, dtype=np.float32)

            print("gt shape: ",gt.shape)
            rgb_img, depth_img, gt = self._augmente_image(rgb_img, depth_img, gt, frame_info.seq_name)
            return rgb_img, depth_img, gt

    def next_batch(self):
        self._scale_ratio = random.uniform(0.7, 1.3)
        self._crop_ratio = random.uniform(0.8, 1)
        self._flip_probability = random.uniform(0, 1)
        print("***** new batch ",self._scale_ratio, self._crop_ratio, self._flip_probability)

    def _augmente_image(self, rgb, depth, gt, seq):
        rgb, offset = utils.crop3d(rgb, self._crop_ratio)
        depth,_ = utils.crop3d(depth, self._crop_ratio, offset)
        gt,_ = utils.crop2d(gt, self._crop_ratio, offset)

        rgb = utils.scale3d(rgb, self._scale_ratio)
        depth = utils.scale3d(depth, self._scale_ratio)
        gt = utils.scale2d(gt, self._scale_ratio, cv2.INTER_NEAREST)

        if seq not in self.flip_seq_for_augmentation:
            flip_p = self._flip_probability
            self.flip_seq_for_augmentation[seq] = flip_p
        else:
            flip_p = self.flip_seq_for_augmentation[seq]
        # print("before flip ",flip_p)
        rgb = utils.flip3d(rgb, flip_p)
        depth = utils.flip3d(depth, flip_p)
        gt = utils.flip2d(gt, flip_p)

        return rgb, depth, gt