import os
import sys
import cv2
import torch
from scipy.misc import imresize
# import shutil
import random
# import tarfile
# import zipfile
import numpy as np

from torch.utils.data import Dataset
#from torchvision.datasets.utils import download_url
from dataloaders import utils
import math

import yaml
"""
Structure of the dataset
# dataset_root
# |-Bootstrapping
# | |-seq_1_name
# | | | |-ROI.bmp
# | | | |-temporalROI.txt
# | | |-input
# | | | |-in000001.png
# | | | |-in000002.png
# | | | |-in000003.png
# | | | |-...png
# | | |-depth
# | | | |-d000001.png
# | | | |-d000002.png
# | | | |-...png
# | | |-groundtruth
# | | | |-gt000138.png
# | | | |-gt000142.png
# | | | |-....png
# | |-seq_2_name
# | | |-...
# | |-seq_...
# |
# |-ColorCamouflage
# | |-seq_1_name
# | | |-...
# | |-seq_2_name
# | | |-...
# | |-seq_...
# |-...

OutOfRange/MultiPeople1:
depth
groundtruth
input
ROI.bmp
temporalROI.txt

OutOfRange/MultiPeople1/depth:
d000001.png
d000002.png
d000003.png
d000004.png
d000005.png
d000006.png
...

OutOfRange/MultiPeople1/groundtruth:
gt000138.png
gt000150.png
gt000162.png
gt000174.png
gt000186.png
...

OutOfRange/MultiPeople1/input:
in000001.png
in000002.png
in000003.png
in000004.png
in000005.png
...

"""

"""
Some of the code were copied from https://github.com/xapharius/pytorch-nyuv2
"""

class SequenceInfo:
    def __init__(self):
        self.root = ""
        self.path_to_rgb = ""
        self.path_to_depth = ""
        self.path_to_gt = ""
        self.names_of_rgb_imgs = []
        self.names_of_depth_imgs = []
        self.names_of_gt_imgs = []
    
    def get_path_to_rgb_img_by_id(self, img_id):
        pass
    
    def get_path_to_rgb_img_by_index(self, idx_of_list):
        pass

    def get_path_to_depth_img_by_id(self, img_id):
        pass

    def get_path_to_depth_img_by_index(self, idx_of_list):
        pass

    def get_path_to_gt_img_by_id(self, img_id):
        pass

    def get_path_to_gt_img_by_index(self, idx_of_list):
        pass


class Folder:
    def __init__(self, folder_name, extension_name=None):
        self.folder_name = folder_name
        self.extension_name = extension_name

    def set_extension_name(self, extension_name):
        self.extension_name = extension_name

class EContentInfo:
    rgb = Folder('input', 'png')
    depth = Folder('depth', 'png')
    groundtruth = Folder('groundtruth', 'png')
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


class sbm_rgbd(Dataset):
    """
    PyTorch wrapper for the SBM-RGBD dataset.
    Data sources available: RGB, Semantic Segmentation, Depth Images.
    If no transformation is provided, the image type will not be returned.
    ### Output
    All images are of size: 640 x 480
    1. RGB: 3 channel input image
    2. Semantic Segmentation: 1 channel representing one of the 14 (0 -
    background) classes. Conversion to int will happen automatically if
    transformation ends in a tensor.
    3. Depth Images: 1 channel with floats representing the distance in meters.
    Conversion will happen automatically if transformation ends in a tensor.
    """

    def __init__(
        self,
        dataset_root,
        sample_range,
        output_HW=None,
        channels_for_target_frame = 'rgbdt',
        channels_for_counterpart_frame = 'rgbdt',
        for_training = True,
        batch_size = 1,
        subset_percentage = 0.8,
        subset = None,
        meanval=(104.00699, 116.66877, 122.67892)
    ):
        """
        Will return tuples based on what data source has been enabled (rgb, seg etc).
        :param root: path to root folder (eg /data/sbm-rgbd)
        :param train: whether to load the train or test set
        """
        self.dataset_root = dataset_root
        self.sample_range = sample_range
        self.output_HW = output_HW # H W
        self.subset_percentage = subset_percentage
        self.meanval = meanval
        self.channels_for_target_frame = channels_for_target_frame
        self.channels_for_counterpart_frame = channels_for_counterpart_frame

        self.batch_size = batch_size
        self.stage = 'train' if for_training else 'test'

        self.flip_prob_of_seqs_for_augmentation = {} # seq_name: flip_probability. During training, images can be flipped horizontally for augmentation. Frames of a sequence should be all flipped or all not flipped.

        self.sets = {
            'entire': {
                'names_of_sequences': [],
                'frame_range_of_sequences': {},
                'names_of_frames': []
            },
            'train': {
                'names_of_sequences': [],
                'frame_range_of_sequences': {},
                'names_of_frames': [] # it is more convenient for iterating frames by organizing all frames of all sequence together 
            },
            'validate': {
                'names_of_sequences': [],
                'frame_range_of_sequences': {},
                'names_of_frames': []
            },
            'test': {
                'names_of_sequences': [],
                'frame_range_of_sequences': {},
                'names_of_frames': []
            }
        }

        self._collect_file_list()
        self._split_dataset(subset)



    def _get_path(self, content_type, seq_name=None, frame_name=None): #(self, content_type:Folder, seq_name=None, frame_name=None):
        if seq_name == None or seq_name not in self.sets['entire']['names_of_sequences']:
            raise Exception('Cannot find sequence ' + seq_name)
        if frame_name == None:
            return os.path.join(self.dataset_root, seq_name, content_type.folder_name)

        # frames_of_seq = self._get_frames_of_seq(seq_name)
        # if frames_of_seq == None:
        #     raise Exception('Cannot find any frame for ' + seq_name)
        # if frame_name not in frames_of_seq:
        #     raise Exception('Frame ' + frame_name + ' is not present in sequence ' + seq_name)

        return os.path.join(self.dataset_root, seq_name, content_type.folder_name, frame_name)

    def _get_path_of_rgb_data(self, seq_name=None, frame_name=None):
        return self._get_path(EContentInfo.rgb, seq_name, frame_name)

    def _get_path_of_depth_data(self, seq_name=None, frame_name=None):
        return self._get_path(EContentInfo.depth, seq_name, frame_name)

    def _get_path_of_groundtruth_data(self, seq_name=None, frame_name=None):
        return self._get_path(EContentInfo.groundtruth, seq_name, frame_name)


    def _collect_file_list(self):
        def __check_framenames_of_sequence(folder, seq): #(folder:Folder, seq):
            seq_path_of_folder = self._get_path(folder, seq)
            if not os.path.exists(seq_path_of_folder):
                # seq doesn't exist
                return None
            framenames_of_seq = os.listdir(seq_path_of_folder)
            return framenames_of_seq
        
        def __get_id_from_framename(name):
            return name[2:-4]
        
        def __get_rgb_framename_from_id(id):
            return 'in'+id+'.png'
        
        def __get_depth_framename_from_id(id):
            return 'd'+id+'.png'
        
        def __get_gt_framename_from_id(id):
            return 'gt'+id+'.png'

        data_types = os.listdir(self.dataset_root)
        for dt in data_types:
            type_path = os.path.join(self.dataset_root, dt)
            seqs = os.listdir(type_path)
            seqs = [os.path.join(dt, seq) for seq in seqs] # put the data type in front of the name of the sqeuence
            self.sets['entire']['names_of_sequences'] = self.sets['entire']['names_of_sequences'] + seqs
        
        invalid_seqs = []
        for seq in self.sets['entire']['names_of_sequences']:
            names_of_rgb_frames_of_seq = __check_framenames_of_sequence(EContentInfo.rgb, seq)
            names_of_depth_frames_of_seq = __check_framenames_of_sequence(EContentInfo.depth, seq)
            names_of_gt_frames_of_seq = __check_framenames_of_sequence(EContentInfo.groundtruth, seq)
            if names_of_gt_frames_of_seq == None or names_of_depth_frames_of_seq == None or names_of_rgb_frames_of_seq == None:
                invalid_seqs.append(seq)
                continue
            names_of_rgb_frames_of_seq.sort()
            names_of_depth_frames_of_seq.sort()
            names_of_gt_frames_of_seq.sort()
            labelled_frames_of_seq = []
            # the name of a ground truth frame is like gtXXXXXX.png, XXXXXX is the id of the gt frame, and it is also the id of a corresponding frame under RGB_data
            ids_of_gt_frames = [__get_id_from_framename(frame) for frame in names_of_gt_frames_of_seq]
            for gt_frame_id in ids_of_gt_frames:
                rgb_framename = __get_rgb_framename_from_id(gt_frame_id)
                assert(rgb_framename in names_of_rgb_frames_of_seq, 'RGB frame '+rgb_framename+' of '+seq+' does not exist.')

                depth_framename = __get_depth_framename_from_id(gt_frame_id)
                assert(depth_framename in names_of_depth_frames_of_seq, 'Depth frame '+depth_framename+' of '+seq+' does not exist.')

                gt_framename = __get_gt_framename_from_id(gt_frame_id)
                assert(gt_framename in names_of_gt_frames_of_seq, 'GT frame '+gt_framename+' of '+seq+' does not exist.')

                videoFrameInfo = VideoFrameInfo(seq, gt_frame_id, rgb_framename, depth_framename, gt_framename)
                labelled_frames_of_seq.append(videoFrameInfo)

            # now all valid frames (labelled frames) of this sequence have been collected in labelled_frames_of_seq
            if len(labelled_frames_of_seq) > 0:
                start_idx = len(self.sets['entire']['names_of_frames'])
                end_idx = start_idx + len(labelled_frames_of_seq)
                self.sets['entire']['frame_range_of_sequences'][seq] = {'start': start_idx, 'end': end_idx}
                self.sets['entire']['names_of_frames'].extend(labelled_frames_of_seq)
        for seq in invalid_seqs:
            self.sets['entire']['names_of_sequences'].remove(seq)

    def _split_dataset(self, predefined_subset_dict):
        if predefined_subset_dict and isinstance(predefined_subset_dict, dict):
            to_be_in_subset = self.stage
            for seq in predefined_subset_dict:
                self.sets[to_be_in_subset]['names_of_sequences'].append(seq)

                start_idx = len(self.sets[to_be_in_subset]['names_of_frames'])
                # get ids of the frames of the sequence
                ids_of_frames = [frame[:2] for frame in predefined_subset_dict[seq]] # We know there are duplicate frames (XX is present more than once)
                # collect information of the frames
                frames_of_seq = []
                for id in ids_of_frames:
                    fn = self._get_framename_of_seq_by_id(seq, id)
                    if fn:
                        frames_of_seq.append(fn)

                num_frames = len(frames_of_seq)
                end_idx = num_frames + start_idx
                self.sets[to_be_in_subset]['frame_range_of_sequences'][seq] = {'start': start_idx, 'end': end_idx}
                self.sets[to_be_in_subset]['names_of_frames'].extend(frames_of_seq)
            return

        else:
            to_be_in_subset = self.stage
            for seq in self.sets['entire']['names_of_sequences']:
                frames_of_seq = self._get_frames_of_seq('entire', seq)
                if not frames_of_seq:
                    raise Exception('Cannot find any frame for '+seq)
                if len(frames_of_seq) < 2 and self.stage == 'train':
                    print("Sequence "+seq+" only has 1 frame. So it cannot be fed for training.")
                    continue

                self.sets[to_be_in_subset]['names_of_sequences'].append(seq) # add this sequence to this set
                start_idx = len(self.sets[to_be_in_subset]['names_of_frames'])
                num_frames = int(math.floor(len(frames_of_seq) * self.subset_percentage))
                if num_frames < 2:
                    if self.stage == 'train':
                        num_frames = 2 # 2 frames at least for a sequence for co-attention, and here the sequence should have more than 1 frame in total
                    # for testing, we can accept a sequence having only 1 frame
                if num_frames == len(frames_of_seq):
                    frames_selected = frames_of_seq
                else:
                    frames_selected = random.sample(frames_of_seq, num_frames)
                end_idx = num_frames + start_idx
                self.sets[to_be_in_subset]['frame_range_of_sequences'][seq] = {'start': start_idx, 'end': end_idx}
                self.sets[to_be_in_subset]['names_of_frames'].extend(frames_selected)


    def _get_frames_of_seq(self, set_name, seq_name):
        seq_offset = self.sets[set_name]['frame_range_of_sequences'][seq_name]
        if seq_offset:
            return self.sets[set_name]['names_of_frames'][seq_offset['start']:seq_offset['end']]
        return None

    def _get_framename_by_index(self, set_name, frame_index):
        if frame_index >= len(self.sets[set_name]['names_of_frames']):
            return None
        return self.sets[set_name]['names_of_frames'][frame_index]

    def __getitem__(self, frame_index):
        set_name = self.stage
        frame_info = self._get_framename_by_index(set_name, frame_index)
        if frame_info:
            sample = {
                'seq_name': frame_info.seq_name, 'frame_index': frame_info.id,
                'target': None, 'target_depth': None, 'target_gt': None, 
                'search_0': None, 'search_0_depth': None, 'search_0_gt':None
            }

            # 1. target frame
            current_rgb, current_depth, current_gt = self._load_images(frame_info, self.channels_for_target_frame)
            sample['target'] = current_rgb
            sample['target_depth'] = current_depth
            sample['target_gt'] = current_gt

            # 2. (a) counterpart frame(s) for comparison with the target frame
            frame_range_of_seq = self.sets[set_name]['frame_range_of_sequences'][frame_info.seq_name]
            frame_indices_for_counterpart = []
            if self.sample_range >= 1:
                frame_indices_candidates= list(range(frame_range_of_seq['start'], frame_range_of_seq['end']))
                frame_indices_for_counterpart = random.sample(frame_indices_candidates, self.sample_range)#min(len(self.img_list)-1, target_id+np.random.randint(1,self.range+1))
            else:
                # use the same frame to the target frame as the matching/search frame
                frame_indices_for_counterpart = [frame_index]

            for i in range(len(frame_indices_for_counterpart)):
                key = 'search_'+str(i)
                frame_idx = frame_indices_for_counterpart[i]
                frame_info_of_cp = self._get_framename_by_index(set_name, frame_idx)
                cp_rgb, cp_depth, cp_gt = self._load_images(frame_info_of_cp, self.channels_for_counterpart_frame)
                sample[key] = cp_rgb
                sample[key+'_depth'] = cp_depth
                sample[key+'_gt'] = cp_gt

            # print(" ##### sample rgb: ",sample['target'].shape, " gt: ", sample['target_gt'].shape,  " depth: ", sample['target_depth'].shape, " search_rgb: " ,sample['search_0'].shape, " search_0_gt: ",sample['search_0_gt'].shape,  "search_0_depth: ", sample['search_0_depth'].shape)
            return sample
        else:
            raise Exception('Cannot find the sequence from frame index ', frame_index)

    def __len__(self):
        set_name = self.stage
        result = len(self.sets[set_name]['names_of_frames'])
        if result % self.batch_size != 0:
            result = result - result % self.batch_size
        print("dataset: ", '  '.join(map(str, self.sets[set_name]['names_of_frames'])))
        print("SBM length: " , result, " for " + set_name)
        return result
    

    def _load_images(self, frame_info, channels='rgbdt'):
        '''
        Load images in channels givin frame_info
        : param frame_info: the info of a frame being loaded
        : param channels: rgbdt stands for red, green, blue, depth, ground truth
        '''

        load_rgb = 'rgb' in channels
        load_depth = 'd' in channels
        load_groundtruth = 't' in channels
        rgb_img = None
        depth_img = None
        gt_img = None
        crop_offset = None
        if load_rgb:
            # load rgb
            rgb_path = self._get_path_of_rgb_data(frame_info.seq_name, frame_info.name_of_rgb_frame)
            if rgb_path:
                rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
                if self.output_HW is not None:
                    rgb_img = imresize(rgb_img, self.output_HW)
                rgb_img = np.array(rgb_img, dtype=np.float32)
                rgb_img = np.subtract(rgb_img, np.array(self.meanval, dtype=np.float32)) 
                rgb_img = rgb_img.transpose((2, 0, 1))  # HWC -> CHW
                if self.stage == 'train':
                    rgb_img, crop_offset = self._augmente_image(rgb_img, frame_info.seq_name, crop_offset, True)
            else:
                raise Exception("Cannot find the rgb image for ", frame_info.seq_name, frame_info.name_of_rgb_frame)
        else:
            rgb_img = np.zeros((1,1), dtype=np.float32)

        if load_depth:
            # load depth
            depth_path = self._get_path_of_depth_data(frame_info.seq_name, frame_info.name_of_depth_frame)
            if depth_path:
                depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
                depth_img = depth_img[None, :,:] # 1, H, W
                if self.stage == 'train':
                    depth_img, crop_offset = self._augmente_image(depth_img, frame_info.seq_name, crop_offset, True)
            else:
                raise Exception("Cannot find the depth image for ", frame_info.seq_name, frame_info.name_of_rgb_frame)
        else:
            depth_img = np.zeros((1,1), dtype=np.float32)

        if load_groundtruth:
            # load gt
            gt_path = self._get_path_of_groundtruth_data(frame_info.seq_name, frame_info.name_of_groundtruth_frame)
            if gt_path:
                gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if self.output_HW is not None:
                    gt_img = imresize(gt_img, self.output_HW, interp='nearest')
                gt_img[gt_img!=0]=1 # H, W with values in {0, 1}
                gt_img = np.array(gt_img, dtype=np.uint8)
                # print("gt shape: ",gt_img.shape)
                if self.stage == 'train':
                    gt_img, crop_offset = self._augmente_image(gt_img, frame_info.seq_name, crop_offset, False)
            else:
                raise Exception("Cannot find the groud truth image for ", frame_info.seq_name, frame_info.name_of_rgb_frame)
        else:
            gt_img = np.zeros((1,1), dtype=np.uint8)

        # to avoid the error `ValueError: some of the strides of a given numpy array are negative. This is currently not supported`
        rgb_img = torch.from_numpy(rgb_img.copy())
        depth_img = torch.from_numpy(depth_img.copy())
        gt_img = torch.from_numpy(gt_img.copy())

        return rgb_img, depth_img, gt_img


    def next_batch(self):
        self._scale_ratio = random.uniform(0.7, 1.3)
        self._crop_ratio = random.uniform(0.8, 1)
        # print("***** new batch ",self._scale_ratio, self._crop_ratio, self._flip_probability)

    def _augmente_image(self, img, seq, offset, is3D):
        # for flipping images, we need to keep the frames of a sequence same, flip all frames of a sequence or not flip any frame of the sequence
        if seq not in self.flip_prob_of_seqs_for_augmentation:
            flip_p = random.uniform(0, 1)
            self.flip_prob_of_seqs_for_augmentation[seq] = flip_p
        else:
            flip_p = self.flip_prob_of_seqs_for_augmentation[seq]

        if is3D:
            img, offset = utils.crop3d(img, self._crop_ratio, offset)
            img = utils.scale3d(img, self._scale_ratio)
            img = utils.flip3d(img, flip_p)
        else:
            img, offset = utils.crop2d(img, self._crop_ratio, offset)
            img = utils.scale2d(img, self._scale_ratio, cv2.INTER_NEAREST)
            img = utils.flip2d(img, flip_p)

        return img, offset
