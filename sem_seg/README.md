## Environment 
### Docker (cuda 11.1)
```
# Pull docker image
docker pull jaesungchoe/pointmixer:cuda11.1

# create your own container and attach to the container.
# (for instance) docker run -it --gpus '"device=0"' --name pointmixer --shm-size 16G --net=host -e NVIDIA_VISIBLE_DEVICES=0 jaesungchoe/pointmixer:cuda11.1
# docker start pointmixer
# docker attach pointmixer
```

### S3DIS 
- TBU (shell)
- result tables

### ScanNet
- TBU (shell)
- result tables
