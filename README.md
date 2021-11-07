An implementation for COMP589: Unsupervised RGBD Video Object Segmentation With Co-attention Siamese Networks.

#### Network Architecture 

![](../master/RGBD Co-attention UVOS.jpg)

#### Requirements

python          3.7
pytorch         1.9.0
opencv          3.4.2
numpy           1.20
pyyaml          5.4.1
scipy           1.7.1
mattplotlib     3.4.2
h5py            2.8.0
torchsummary    1.4.5

#### Dataset
1. Download the SBM-RGBD dataset
    wget https://rgbd2017.na.icar.cnr.it/SBM-RGBDdataset/IlluminationChanges/IlluminationChanges.zip
    wget https://rgbd2017.na.icar.cnr.it/SBM-RGBDdataset/ColorCamouflage/ColorCamouflage.zip
    wget https://rgbd2017.na.icar.cnr.it/SBM-RGBDdataset/DepthCamouflage/DepthCamouflage.zip
    wget https://rgbd2017.na.icar.cnr.it/SBM-RGBDdataset/IntermittentMotion/IntermittentMotion.zip
    wget https://rgbd2017.na.icar.cnr.it/SBM-RGBDdataset/OutOfRange/OutOfRange.zip
    wget https://rgbd2017.na.icar.cnr.it/SBM-RGBDdataset/Shadows/Shadows.zip
    wget https://rgbd2017.na.icar.cnr.it/SBM-RGBDdataset/Bootstrapping/Bootstrapping.zip

2. Unzip the downloaded dataset zips to a folder by following the structure described in dataset_info/sbm-rgbd-file-list.txt. And unzip dataset_info/ROIs.zip to the corresponding folders in the dataset.

3. Search all "data_path" under "sbmrgbd" in config.yaml and replace the their values with the path to the dataset folder from step 1.

#### Testing

1. Install libraries/frameworks from the list of the requirements.

2. Prepare the dataset by following descriptions in "Dataset" section.

3. Download the pretrained model from . And change the value of "test/model/resnet_aspp_add/pretrained_params" to path to the file.

4. Run 'test.py --dataset sbmrgbd --model raa --gpus X[,Y]'. X, Y are the GPU number of your graphics card. For example, 'python test.py --dataset sbmrgbd --model raa --gpus 0,1'.

#### Training

1. Install libraries/frameworks from the list of the requirements.

2. Prepare the dataset by following descriptions in "Dataset" section.

3. Download the pretrained model from . And change the value of "test/model/resnet_aspp_add/pretrained_params" to path to the file.

4. Download the deeplabv3 model from [GoogleDrive](https://drive.google.com/open?id=1hy0-BAEestT9H4a3Sv78xrHrzmZga9mj). Put it into the folder pretrained/deep_labv3.

5. Change the video path, saliency dataset path and deeplabv3 path in config.yaml.
The folder of DAVIS dataset is like
![image](https://user-images.githubusercontent.com/11287531/116809350-af9f6a80-ab91-11eb-9ae0-88a3cfb1243b.png)
The folder of saliency dataset is like
![image](https://user-images.githubusercontent.com/11287531/116809415-073dd600-ab92-11eb-93a9-3eff05bd193f.png)


4. Run command: python train_iteration_conf.py --dataset davis --gpus 0,1

### Citation

If you find the code and dataset useful in your research, please consider citing:
```
@InProceedings{Lu_2019_CVPR,  
author = {Lu, Xiankai and Wang, Wenguan and Ma, Chao and Shen, Jianbing and Shao, Ling and Porikli, Fatih},  
title = {See More, Know More: Unsupervised Video Object Segmentation With Co-Attention Siamese Networks},  
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},  
year = {2019}  
}
@article{lu2020_pami,
  title={Zero-Shot Video Object Segmentation with Co-Attention Siamese Networks},
  author={Lu, Xiankai and Wang, Wenguan and Shen, Jianbing and Crandall, David and Luo, Jiebo},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2020},
  publisher={IEEE}
}
```
### Other related projects/papers:
[Saliency-Aware Geodesic Video Object Segmentation (CVPR15)](https://github.com/wenguanwang/saliencysegment)

[Learning Unsupervised Video Primary Object Segmentation through Visual Attention (CVPR19)](https://github.com/wenguanwang/AGS)

Any comments, please email: carrierlxk@gmail.com
