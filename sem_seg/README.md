# 3D Semantic Segmentation 
<img src="./fig/semseg.JPG" width="534" height="222"> <br/>
This sub-repository includes the implementation of the baselines with the S3DIS and ScanNet datasets.
- PointMixer, ECCV 2022 (ours)
- PointTransformer, ICCV 2021 

## Features
- Can process arbitrary length of batch size, [COO format](https://nvidia.github.io/MinkowskiEngine/terminology.html?highlight=coo%20format#sparse-tensor).<br/>
  - No longer use ~~[batch, kNN, channel]~~.<br/>
- Can reproduce the results using less GPU memory.
  - 1 x 3090Ti GPU or 2 x 1080Ti GPU (24GB)
- Plug-and-play Implementation, starting from training to evaluation.
  - 42 hours for training and 2 hours for evaluation  (~44 hours in total).
  

## Implementation
### Docker (cuda 11.1)
```
# Pull docker image
docker pull jaesungchoe/pointmixer:cuda11.1

# create your own container and attach to the container
docker run -it --gpus '"device=0"' --name pointmixer --shm-size 16G \
  --net=host -e NVIDIA_VISIBLE_DEVICES=0 jaesungchoe/pointmixer:cuda11.1
docker start pointmixer
docker attach pointmixer

# dataset prepration 

# Train / Test
cd /code/ECCV22-PointMixer/sem_seg
sh script/run_s3dis_PointMixer.sh 
sh script/run_scannet_PointMixer.sh 
```
### Conda
```
# load conda env
cd ./conda
conda env create -f environment.yml

# dataset prepration 

# Train / Test
git clone https://github.com/LifeBeyondExpectations/ECCV22-PointMixer
cd ./ECCV22-PointMixer/sem_seg
sh script/run_s3dis_PointMixer.sh
sh script/run_scannet_PointMixer.sh 
```

## Quantitative results
### S3DIS Area5 test
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pointmixer-mlp-mixer-for-point-cloud/semantic-segmentation-on-s3dis-area5)](https://paperswithcode.com/sota/semantic-segmentation-on-s3dis-area5?p=pointmixer-mlp-mixer-for-point-cloud)
| Voxel Size | Model                |       Params |         mAcc |         mIoU | Reference |
|:----------:|:---------------------|-------------:|-------------:|-------------:|:---------:|
|        2cm | MinkowskiNet42       |        37.9M |         74.1 |         67.2 | [Checkpoint](https://postechackr-my.sharepoint.com/:u:/g/personal/p0125ch_postech_ac_kr/EZcO0DH6QeNGgIwGFZsmL-4BAlikmHAHlBs4JBcS5XfpVQ?download=1) |
|            | FastPointTransformer |        37.9M | :+1:**77.6** |         71.0 | [github](https://github.com/POSTECH-CVLab/FastPointTransformer) |
|            | PointTransformer     |         7.8M |         76.5 |         70.4 | [Codes from the authors](https://github.com/POSTECH-CVLab/point-transformer) |
|            | PointMixer (ours)    | :+1:**6.5M** |         77.4 | :+1:**71.4** | TBU |

### ScanNet V2 validation
| Voxel Size | Model                | mAcc | mIoU | Reference |
|:----------:|:---------------------|:----:|:----:|:---------:|
|       10cm | MinkowskiNet42       | 70.8 | 60.7 | [Official GitHub](https://github.com/chrischoy/SpatioTemporalSegmentation) |
|            | FastPointTransformer | 76.1 | 66.5 | - |
|            | PointTransformer     |    - |    - | - |
|            | PointMixer (ours)    |    - |    - | - |
|        5cm | MinkowskiNet42       | 76.3 | 67.0 | [Official GitHub](https://github.com/chrischoy/SpatioTemporalSegmentation) |
|            | FastPointTransformer | 78.9 | 70.0 | - |
|            | PointTransformer     |    - |    - | - |
|            | PointMixer (ours)    |    - |    - | - |
|        2cm | MinkowskiNet42       | 80.4 | 72.2 | [Official GitHub](https://github.com/chrischoy/SpatioTemporalSegmentation) |
|            | FastPointTransformer | 81.2 | 72.5 | - |
|            | PointTransformer     |    - |    - | - |
|            | PointMixer (ours)    |    - |    - | - |

## Dataset preparation
### S3DIS dataset
- Download data (Please refer to [PAConv](https://github.com/CVMI-Lab/PAConv/tree/main/scene_seg#dataset)).
- Update the dataroot in `script/run_s3dis_PointMixer.sh`.

### ScanNet dataset
- Download data (Please refer to [Stratified-Transformer](https://github.com/dvlab-research/Stratified-Transformer#scannetv2)).
- Update the dataroot in `script/run_scannet_PointMixer.sh`.

### Resulting dataset structure
```
${data_dir}
├── s3dis
│   ├── list
│   │   ├── s3dis_names.txt
│   │   ├── val5.txt
│   │   └── ...
│   ├── trainval
│   │   ├── 00001071.h5
│   │   └── ...
│   └── trainval_fullarea
│       ├── Area_1_conferenceRoom_1.npy
│       └── ...
└── scannet_semseg
    ├── test
    │   ├── scene0707_00_inst_nostuff.pth
    │   └── ...
    ├── train
    │   ├── scene0000_00_inst_nostuff.pth
    │   └── ...
    └── val
        ├── scene0011_00_inst_nostuff.pth
        └── ...
```
