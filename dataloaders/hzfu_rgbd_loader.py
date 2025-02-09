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
# from scipy.misc import imresize
from torch.utils.data import Dataset
from dataloaders import utils
import torch
import math
from PIL import Image

k_method_of_splitting_dataset = 'frame_in_out' #'sequence_in_out'

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
                 output_HW=None,
                 channels_for_target_frame = 'rgbdt',
                 channels_for_counterpart_frame = 'rgbdt',
                 for_training = True,
                 subset_percentage = 0.8,
                 subset = None,
                 batch_size = 1,
                 meanval=(104.00699, 116.66877, 122.67892),
                 transform=None,
                 output_dir_for_debug=None
                 ):
        self.dataset_root = dataset_root
        self.sample_range = sample_range
        self.output_HW = output_HW # H W
        self.subset_percentage = subset_percentage
        self.transform = transform
        self.meanval = meanval
        self.stage = 'train' if for_training else 'test'
        self.channels_for_target_frame = channels_for_target_frame
        self.channels_for_counterpart_frame = channels_for_counterpart_frame
        self.depth_min_max = {}
        self.output_dir_for_debug = output_dir_for_debug

        self.flip_prob_of_seqs_for_augmentation = {} # seq_name: flip_probability. During training, images can be flipped horizontally for augmentation. Frames of a sequence should be all flipped or all not flipped.

        self.sets = {
            'entire': {
                'names_of_sequences': [],
                'frame_range_of_sequences': {},
                'names_of_frames': [],
                'frame_to_sequence': []
            },
            'train': {
                'names_of_sequences': [],
                'frame_range_of_sequences': {},
                'names_of_frames': [], # it is more convenient for iterating frames by organizing all frames of all sequence together 
                'frame_to_sequence': []
            },
            'validate': {
                'names_of_sequences': [],
                'frame_range_of_sequences': {},
                'names_of_frames': [],
                'frame_to_sequence': []
            },
            'test': {
                'names_of_sequences': [],
                'frame_range_of_sequences': {},
                'names_of_frames': [],
                'frame_to_sequence': []
            }
        }

        self.batch_size = batch_size

        self._load_meta_data()
        self._split_dataset(subset)


    def new_training_epoch(self):
        self.flip_prob_of_seqs_for_augmentation.clear()

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
        seq_offset = self.sets[set_name]['frame_range_of_sequences'][seq_name]
        if seq_offset:
            return self.sets[set_name]['names_of_frames'][seq_offset['start']:seq_offset['end']]
        return None

    def _get_framename_by_index(self, set_name, frame_index):
        if frame_index >= len(self.sets[set_name]['names_of_frames']):
            return None
        return self.sets[set_name]['names_of_frames'][frame_index]

    def _get_framename_of_seq_by_id(self, seq, id):
        for fn in self.sets['entire']['names_of_frames']:
            if fn.id == id and fn.seq_name == seq:
                return fn
        return None

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
            if not os.path.exists(seq_path):
                # seq doesn't exist
                return None
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

        invalid_seqs = []
        for seq in self.sets['entire']['names_of_sequences']:
            '''
            because not every rgb frame has been labelled in the dataset and I only consider the labelled frames as valid frames for training and test,
            I need to collect rgb frames that have corresponding grountruth frame
            '''
            names_of_rgb_frames_of_seq = __check_framenames_of_sequence(EContentInfo.rgb, seq)
            names_of_depth_frames_of_seq = __check_framenames_of_sequence(EContentInfo.depth, seq)
            names_of_gt_frames_of_seq = __check_framenames_of_sequence(EContentInfo.groundtruth, seq)

            if names_of_gt_frames_of_seq == None or names_of_depth_frames_of_seq == None or names_of_rgb_frames_of_seq == None:
                invalid_seqs.append(seq)
                continue

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

        
        if k_method_of_splitting_dataset == 'sequence_in_out':
            for seq in self.sets['entire']['names_of_sequences']:
                # sequence based -- all frames of a sequence is randomly chosen for training, or for test
                rand = random.random()
                frames_of_seq = self._get_frames_of_seq('entire', seq)
                if not frames_of_seq:
                    raise Exception('Cannot find any frame for '+seq)

                if rand < self.subset_percentage:
                    to_be_in_subset = self.stage
                else:
                    to_be_in_subset = 'test' if self.stage == 'train' else 'train'

                seq_index = len(self.sets[to_be_in_subset]['names_of_sequences'])
                self.sets[to_be_in_subset]['names_of_sequences'].append(seq) # add this sequence to this set
    
                start_idx = len(self.sets[to_be_in_subset]['names_of_frames'])
                num_frames = len(frames_of_seq)
                end_idx = num_frames + start_idx
                self.sets[to_be_in_subset]['frame_range_of_sequences'][seq] = {'start': start_idx, 'end': end_idx}
                self.sets[to_be_in_subset]['names_of_frames'].extend(frames_of_seq)
                #self.sets[to_be_in_subset]['frame_to_sequence'].extend([seq_index]*num_frames)
        elif k_method_of_splitting_dataset == 'frame_in_out':
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
                num_frames = math.floor(len(frames_of_seq) * self.subset_percentage)
                if num_frames < 2:
                    if self.stage == 'train':
                        num_frames = 2 # 2 frames at least for a sequence for co-attention, and here the sequence should have more than 1 frame in total
                    # for testing, we can accept a sequence having only 1 frame
                if num_frames == len(frames_of_seq):
                    frames_selected = frames_of_seq
                else:
                    frames_selected = random.sample(frames_of_seq, num_frames)
                end_idx = (int)(num_frames + start_idx)
                self.sets[to_be_in_subset]['frame_range_of_sequences'][seq] = {'start': start_idx, 'end': end_idx}
                self.sets[to_be_in_subset]['names_of_frames'].extend(frames_selected)


    # implementation
    def __len__(self):
        set_name = self.stage
        result = len(self.sets[set_name]['names_of_frames'])
        if result % self.batch_size != 0:
            result = result - result % self.batch_size
        print("dataset: ", '  '.join(map(str, self.sets[set_name]['names_of_frames'])))
        print("HzFuRGBDVideos length: " , result, " for " + set_name)
        return result

    def __getitem__(self, frame_index):
        '''
        :return: the rgb, depth of the target frame and the rgb, depth of the matching/search frame
        '''
        def _load_frame(frame_info, channels_to_load):

            def _use_depth_as_rgb(depth_data):
                '''
                :return: a tensor in the shape of (3, rows, columns), every channel of the tensor is a copy of the depth_data subracting the mean value
                '''
                r = np.array(depth_data).copy()
                g = r.copy()
                b = g.copy()
                rgb = np.stack((r,g,b), axis=2) # (rows, columns, 3 channels)
                rgb = np.subtract(rgb, np.array(self.meanval, dtype=np.float32))
                rgb = rgb.transpose((2,0,1)) # from (rows, columns, channels) to (channels, rows, columns)
                return rgb

            rgb, depth, gt = self._load_images(frame_info, channels_to_load)
            if 'rgb' not in channels_to_load:
                if 'd' in channels_to_load:
                    rgb = _use_depth_as_rgb(depth[0])
                else:
                    raise Exception("Invalid 'channels' parameter, which should be 'd' or 'rgb' or 'rgbd'.")
            
            if self.output_dir_for_debug != None:
                save_dir = os.path.join(self.output_dir_for_debug, frame_info.seq_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                filename = os.path.join(save_dir, '{}_rgb.png'.format(frame_info.id))
                if type(rgb) != np.ndarray:
                    rgb_npary = rgb.numpy()
                else:
                    rgb_npary = rgb
                img = Image.fromarray(np.uint8(np.add(rgb_npary.transpose(1, 2, 0), self.meanval)), 'RGB') #(rows, columns, channels)
                img.save(filename)

                if 'd' in channels_to_load:
                    filename = os.path.join(save_dir, '{}_depth.png'.format(frame_info.id))
                    img = Image.fromarray(np.uint8(depth[0]), 'L')
                    img.save(filename)

                if 't' in channels_to_load:
                    filename = os.path.join(save_dir, '{}_gt.png'.format(frame_info.id))
                    img = Image.fromarray(np.uint8(gt*255), 'L')
                    img.save(filename)
                del img
            
            return rgb, depth, gt


        set_name = self.stage
        frame_info = self._get_framename_by_index(set_name, frame_index)
        if frame_info:
            sample = {
                'seq_name': frame_info.seq_name, 'frame_index': frame_info.id,
                'target': None, 'target_depth': None, 'target_gt': None, 
                'search_0': None, 'search_0_depth': None, 'search_0_gt':None
            }

            # 1. target frame
            current_rgb, current_depth, current_gt = _load_frame(frame_info, self.channels_for_target_frame)
            sample['target'] = current_rgb
            sample['target_depth'] = current_depth
            sample['target_gt'] = current_gt

            print("Load target frame ", frame_info.id, " of seq: ",frame_info.seq_name) # with this debug infor, we can know which data impact the loss at different extent


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
                cp_rgb, cp_depth, cp_gt = _load_frame(frame_info_of_cp, self.channels_for_counterpart_frame)
                sample[key] = cp_rgb
                sample[key+'_depth'] = cp_depth
                sample[key+'_gt'] = cp_gt
                print(i, " Load search frame ", frame_info_of_cp.id, " of seq: ",frame_info_of_cp.seq_name) # with this debug infor, we can know which data impact the loss at different extent

            # print(" ##### sample rgb: ",sample['target'].shape, " gt: ", sample['target_gt'].shape,  " depth: ", sample['target_depth'].shape, " search_rgb: " ,sample['search_0'].shape, " search_0_gt: ",sample['search_0_gt'].shape,  "search_0_depth: ", sample['search_0_depth'].shape)
            return sample
            # return current_img, current_depth, current_img_gt, match_img, match_depth, match_img_gt

        else:
            raise Exception('Cannot find the sequence from frame index ', frame_index)

    def _load_images(self, frame_info, channels='rgbdt'):
        '''
        Load images in channels givin frame_info
        : param frame_info: the info of a frame being loaded
        : param channels: rgbdt stands for red, green, blue, depth, ground truth
        '''

        def __load_mat(path): #->np.array:
            # in the shape of (H, W) with values in [0, 255]
            '''
            : return: a 2d array in the shape of (output_HW[0], outputHW[1]) with values in [0,255]
            '''
            f = h5py.File(path, 'r')
            result = np.array(f['depth'], dtype=np.float32) # need to be transposed. An image in the size of w:640, h:480, will get w:480, h:640 (transposed, but reshaped) here
            result = result.transpose() # now it is in the same shape with the original image

            # resize to the expected size
            if self.output_HW is not None:
                # result = imresize(result, self.output_HW)
                result = cv2.resize(result,(self.output_HW[1], self.output_HW[0]),interpolation = cv2.INTER_NEAREST)
            result = np.array(result, dtype=np.float32)

            depth_min = result.min()
            depth_max = result.max()

            # normalize
            result = (result - depth_min) * 255 / (depth_max - depth_min)
            # print(" after depth shape: ",result.shape, result.dtype)
            return result, depth_min, depth_max

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
                    rgb_img = cv2.resize(rgb_img,(self.output_HW[1], self.output_HW[0]))

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
                depth_img, depth_min, depth_max = __load_mat(depth_path)

                # img = Image.fromarray(np.uint8(depth_img), 'L')
                # img.save(depth_img_filename)

                if frame_info.seq_name not in self.depth_min_max:
                    self.depth_min_max[frame_info.seq_name] = [depth_min, depth_max]
                else:
                    if self.depth_min_max[frame_info.seq_name][0] < depth_min:
                        self.depth_min_max[frame_info.seq_name][0] = depth_min
                    if self.depth_min_max[frame_info.seq_name][1] > depth_max:
                        self.depth_min_max[frame_info.seq_name][1] = depth_max

                depth_img = depth_img[None, :,:] # 1, H, W with values in [0, 255]
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
                    # gt_img = imresize(gt_img, self.output_HW, interp='nearest')
                    gt_img = cv2.resize(gt_img,(self.output_HW[1], self.output_HW[0]),interpolation = cv2.INTER_NEAREST)

                    
                gt_img[gt_img!=0]=1 # H, W with values in {0, 1}
                gt_img = np.array(gt_img, dtype=np.uint8)
                # print("gt shape: ",gt_img.shape)
                if self.stage == 'train':
                    gt_img, crop_offset = self._augmente_image(gt_img, frame_info.seq_name, crop_offset, False)
            else:
                raise Exception("Cannot find the groud truth image for ", frame_info.seq_name, frame_info.name_of_rgb_frame)
        else:
            gt_img = np.zeros((1,1), dtype=np.uint8)

        # if self.stage == 'train':
        #     rgb_img, depth_img, gt_img = self._augmente_images(rgb_img, depth_img, gt_img, frame_info.seq_name)

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


    def _augmente_images(self, rgb, depth, gt, seq):
        # for flipping images, we need to keep the frames of a sequence same, flip all frames of a sequence or not flip any frame of the sequence
        if seq not in self.flip_prob_of_seqs_for_augmentation:
            flip_p = random.uniform(0, 1)
            self.flip_prob_of_seqs_for_augmentation[seq] = flip_p
        else:
            flip_p = self.flip_prob_of_seqs_for_augmentation[seq]
        # print("before flip ",flip_p)

        offset = None
        if rgb:
            rgb, offset = utils.crop3d(rgb, self._crop_ratio, offset)
            rgb = utils.scale3d(rgb, self._scale_ratio)
            rgb = utils.flip3d(rgb, flip_p)

        if depth:
            depth, offset = utils.crop3d(depth, self._crop_ratio, offset)
            depth = utils.scale3d(depth, self._scale_ratio)
            depth = utils.flip3d(depth, flip_p)

        if gt:
            gt, offset = utils.crop2d(gt, self._crop_ratio, offset)
            gt = utils.scale2d(gt, self._scale_ratio, cv2.INTER_NEAREST)
            gt = utils.flip2d(gt, flip_p)

        return rgb, depth, gt