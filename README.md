# PointMixer: MLP-Mixer for Point Cloud Understanding

Official pytorch implementation for "PointMixer" (ECCV 2022)  
[Jaesung Choe*](https://sites.google.com/view/jaesungchoe), [Chunghyun Park*](https://chrockey.github.io/), [Francois Rameau](https://rameau-fr.github.io/), [Jaesik Park](https://jaesik.info/), and [In So Kweon](https://rcv.kaist.ac.kr).  
(*: equal contribution)

[Paper](https://arxiv.org/pdf/2111.11187) [Project page] [Youtube] [PPT]

<img src="./etc/teaser.jpg" width="300" height="300"> 

- **All the implementation will be released soon.**

## Environment
We provide the ways of building the environment, docker and conda.  

### Docker (cuda 10.2)
```
# Pull docker image
docker pull jaesungchoe/pointmixer:0.1

# create your own container and attach to the container.
# (for instance) docker run -it --gpus '"device=0,1"' --name pointmixer --shm-size 32G --net=host -e NVIDIA_VISIBLE_DEVICES=0,1 jaesungchoe/pointmixer:0.1
# docker start pointmixer
# docker attach pointmixer
# cd /root/

# pull the code in your container
git pull https://github.com/LifeBeyondExpectations/PointMixer/
cd PointMixer

# compile CUDA binary 
sh env_setup.sh
```

### Docker (cuda 11.1)
```
# Pull docker image
docker pull jaesungchoe/pointmixer:cuda11.1

# create your own container and attach to the container.
# (for instance) docker run -it --gpus '"device=0,1"' --name pointmixer --shm-size 32G --net=host -e NVIDIA_VISIBLE_DEVICES=0,1 jaesungchoe/pointmixer:cuda11.1
# docker start pointmixer
# docker attach pointmixer
# cd /root/

# pull the code in your container
git pull https://github.com/LifeBeyondExpectations/PointMixer/
cd PointMixer

# compile CUDA binary 
sh env_setup.sh
```

### Conda environment
(under construction)

## Training / Evaluation
(Please refer to the task-specific guidances in each folder)

## Results

## References
```
@article{choe2021pointmixer,
  title={PointMixer: MLP-Mixer for Point Cloud Understanding},
  author={Choe, Jaesung and Park, Chunghyun and Rameau, Francois and Park, Jaesik and Kweon, In So},
  journal={arXiv preprint arXiv:2111.11187},
  year={2021}
}
```
