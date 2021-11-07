An implementation for COMP589: Unsupervised RGBD Video Object Segmentation With Co-attention Siamese Networks.

#### Network Architecture 

![](./RGBD Co-attention UVOS.jpg)

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

3. Download the pretrained model file from https://drive.google.com/file/d/1d7hpX_w9bQCpn-w1hBH89DsNw1pE5N0x. And change the value of "test/model/resnet_aspp_add/pretrained_params" in config.yaml to the path to this file.

4. Run 'test.py --dataset sbmrgbd --model raa --gpus X[,Y]'. X, Y are the GPU number of your graphics card. For example, 'python test.py --dataset sbmrgbd --model raa --gpus 0,1'.

#### Training

1. Install libraries/frameworks from the list of the requirements.

2. Prepare the dataset by following descriptions in "Dataset" section.

3. Download the pretrained model file from . And change the value of "train/model/resnet_aspp_add/initial_params" in config.yaml to the path to this file.

4. Run command: 'python train.py --dataset sbmrgbd --gpus X[,Y]' X, Y are the GPU number of your graphics card. For example, 'python train.py --dataset sbmrgbd --model raa --gpus 0,1'.

Note: Change the value of "sbmrgbd/subset" to train or test from different subsets.
