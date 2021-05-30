import os
import sys
import h5py
import torch
import shutil
import random
import tarfile
import zipfile
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
#from torchvision.datasets.utils import download_url

import yaml

"""
Some of the code were copied from https://github.com/xapharius/pytorch-nyuv2
"""
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
        root: str,
        full_path_to_config: str,
        train: bool = True,
        rgb_transform=None,
        seg_transform=None,
        depth_transform=None,
    ):
        """
        Will return tuples based on what data source has been enabled (rgb, seg etc).
        :param root: path to root folder (eg /data/sbm-rgbd)
        :param train: whether to load the train or test set
        :param rgb_transform: the transformation pipeline for rbg images
        :param seg_transform: the transformation pipeline for segmentation images. If
        the transformation ends in a tensor, the result will be automatically
        converted to int in [0, 14)
        :param depth_transform: the transformation pipeline for depth images. If the
        transformation ends in a tensor, the result will be automatically converted
        to meters
        """
        super().__init__()
        self.root = root
        self.path_to_config = full_path_to_config

        self.rgb_transform = rgb_transform
        self.seg_transform = seg_transform
        self.depth_transform = depth_transform

        self.train = train
        self._split = "train" if self.train else "test"

        self.filepaths_rgb = [[[]]]
        self.filepaths_

        if os.path.exists(self.path_to_config):
            with open(self.path_to_config) as config_file:
                self.config = yaml.load(config_file)
                self._check_exists()




        # rgb folder as ground truth
        self._files = os.listdir(os.path.join(root, f"{self._split}_rgb"))


    def _collect_file_list(self):

        for vname in self.config[self._split]['dataset']['sbm_rgbd']['names_of_videos']:
            for type_ in ["input", "depth", "groundtruth"]:
                path = os.path.join(self.root, vname, type_)
                if not os.path.exists(path):
                    raise FileNotFoundError("Cannot find folder ", path)
                


    def _check_exists(self) -> bool:
        """
        Only checking for folder existence
        """
        for vname in self.config[self._split]['dataset']['sbm_rgbd']['names_of_videos']:
            for type_ in ["input", "depth", "groundtruth"]:
                path = os.path.join(self.root, vname, type_)
                if not os.path.exists(path):
                    raise FileNotFoundError("Cannot find folder ", path)
