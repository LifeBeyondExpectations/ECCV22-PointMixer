# 3D Semantic Segmentation 
This sub-repository includes the implementation of the three baselines with the S3DIS and ScanNet datasets.
- PointMixer, ECCV 2021 (ours)
- PointTransformer, ICCV 2021
- PointNet++, Neurips 2017



## Environments 
### Docker (cuda 11.1)
```
# Pull docker image.
docker pull jaesungchoe/pointmixer:cuda11.1

# create your own container and attach to the container.
docker run -it --gpus '"device=0"' --name pointmixer --shm-size 16G \
  --net=host -e NVIDIA_VISIBLE_DEVICES=0 jaesungchoe/pointmixer:cuda11.1
docker start pointmixer
docker attach pointmixer
```
### Conda
```
conda env create -f environment.yml
```



## Dataset preparation
### S3DIS (TBU)
### ScanNet (TBU)



## Run
### Train
### Test

## Quantitative results
### S3DIS (TBU)
### ScanNet (TBU)
