# https://hzfu.github.io/proj_rgbdseg.html
# The dataset for Foreground Segmentation in RGBD Video

# Please cite the following publications if you used this work:

# [1] Huazhu Fu, Dong Xu, Stephen Lin, and Jiang Liu, "Object-based RGBD Image Co-segmentation with Mutex Constraint," in CVPR, 2015, pp. 4428–4436.
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
import random
from torch.utils.data import Dataset


k_sub_set_percentage = {
    'train': 0.8,
    'test': 0.2
}

class Folder:
    def __init__(self, folder_name, extension_name):
        self.folder_name = folder_name
        self.extension_name = extension_name

class EContentInfo:
    rgb = Folder('RGB_data', 'png')
    depth = Folder('Depth_data', 'mat')
    groundtruth = Folder('Label', 'png')
    all_folders = [rgb, depth, groundtruth]

class HzFuRGBDVideos(Dataset):
    def __init__(self, dataset_root, 
                 desired_input_size=None,
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892)):
        self.dataset_root = dataset_root
        self.desired_input_size = desired_input_size
        self.transform = transform
        self.meanval = meanval
        self.entire_set = {
            'names_of_sequences': [],
            'offset_of_sequences': {},
            'all_frames': [],
        }

        self.subsets = {
            'train': {
                'names_of_sequences': [],
                'offset_of_sequences': {},
                'all_frames': [],
            },
            'validate': {
                'names_of_sequences': [],
                'offset_of_sequences': {},
                'all_frames': [],
            },
            'test': {
                'names_of_sequences': [],
                'offset_of_sequences': {},
                'all_frames': [],
            }
        }

        self._load_meta_data()
        self._split_dataset()

    def _get_path(self, content_type:Folder, seq_name=None, frame_name=None):
        if seq_name == None:
            return os.path.join(self.dataset_root, content_type.folder_name)
        if seq_name not in self.entire_set['names_of_sequences']:
            raise Exception('Cannot find sequence ' + seq_name)
        if frame_name == None:
            return os.path.join(self.dataset_root, content_type.folder_name, seq_name)
        frames_of_seq = self._get_frames_of_seq(seq_name)
        if frames_of_seq == None:
            raise Exception('Cannot find any frame for ' + seq_name)
        if frame_name not in frames_of_seq:
            raise Exception('Frame ' + frame_name + ' is not present in sequence ' + seq_name)
        return os.path.join(self.dataset_root, content_type.folder_name, seq_name, frame_name, content_type.extension_name)

    def _get_path_of_rgb_data(self, seq_name=None, frame_name=None):
        return self._get_path(EContentInfo.rgb, seq_name, frame_name)

    def _get_path_of_depth_data(self, seq_name=None, frame_name=None):
        return self._get_path(EContentInfo.depth, seq_name, frame_name)

    def _get_path_of_groundtruth_data(self, seq_name=None, frame_name=None):
        return self._get_path(EContentInfo.groundtruth, seq_name, frame_name)

    def _get_frames_of_seq(self, seq_name):
        seq_offset = self.entire_set['offset_of_sequences'][seq_name]
        if seq_offset:
            return self.entire_set['all_frames'][seq_offset['start']: seq_offset['end']]
        return None

    def _load_meta_data(self):
        rgb_data_path = self._get_path_of_rgb_data()
        self.entire_set['names_of_sequences'] = os.listdir(rgb_data_path)

        def __remove_extension_name(fullname:str, known_ext):
            ext_starts = fullname.rindex('.')
            if ext_starts > 0:
                ext = fullname[ext_starts+1:]
                if known_ext and ext != known_ext:
                    raise Exception('Encounter a different extension name '+ ext + ' from ' + fullname)
                return fullname[:ext_starts]
            raise Exception(fullname, ' does not have an extension name.')

        for seq in self.entire_set['names_of_sequences']:
            folder = EContentInfo.rgb
            seq_path = self._get_path(folder, seq)
            names_of_frames_of_seq = os.listdir(seq_path).sort()
            if names_of_frames_of_seq and len(names_of_frames_of_seq) > 0:
                start_idx = len(self.entire_set.all_frames)
                end_idx = start_idx + len(names_of_frames_of_seq)
                self.entire_set['offset_of_sequences'][seq] = {'start': start_idx, 'end': end_idx}
                names_of_frames_of_seq = [__remove_extension_name(itm, folder.extension_name) for itm in names_of_frames_of_seq]
                self.entire_set['all_frames'].extend(names_of_frames_of_seq)

    def _split_dataset(self):
        # sequence based
        seq_num = len(self.entire_set['names_of_sequences'])
        for seq in self.entire_set['names_of_sequences']:
            rand = random.random()
            frames_of_seq = self._get_frames_of_seq(seq)
            if not frames_of_seq:
                raise Exception('Cannot find any frame for '+seq)

            if rand < k_sub_set_percentage.train:
                # train
                to_be_in_subset = 'train'
            else:
                # test
                to_be_in_subset = 'test'

            self.subsets[to_be_in_subset]['names_of_sequences'].append(seq)
            start_idx = len(self.subsets[to_be_in_subset]['all_frames'])
            end_idx = len(frames_of_seq) + start_idx
            self.subsets[to_be_in_subset]['offset_of_sequences'][seq] = {'start': start_idx, 'end': end_idx}
            self.subsets[to_be_in_subset]['all_frames'].extend(frames_of_seq)

    # implementation
    def __len__(self):
        print("HzFuRGBDVideos length: " + len(self.subsets['train']['all_frames']))
        return len(len(self.subsets['train']['all_frames']))

    def __getitem__(self, idx):
        





