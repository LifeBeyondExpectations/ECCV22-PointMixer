# PointMixer: MLP-Mixer for Point Cloud Understanding

This is an official implementation for the paper,
> [PointMixer: MLP-Mixer for Point Cloud Understanding](https://arxiv.org/pdf/2111.11187)<br/>
> [Jaesung Choe*](https://sites.google.com/view/jaesungchoe), [Chunghyun Park*](https://chrockey.github.io/), [Francois Rameau](https://rameau-fr.github.io/), [Jaesik Park](https://jaesik.info/), and [In So Kweon](https://rcv.kaist.ac.kr)<br/>
> European Conference on Computer Vision (ECCV), Tel Aviv, Israel, 2022<br/>
> [Paper](https://arxiv.org/pdf/2111.11187) [Project] [YouTube] [PPT] [Dataset]<br/>

(*: equal contribution)

## We are currently updating this repository :fire:
- [ ] semseg<br/>  
  - [ ] methods
    - [x] ~~pointmixer~~
    - [ ] point transformer
    - [ ] pointnet++
  - [ ] s3dis weights  
  - [ ] scannet weights
- [ ] objcls<br/>
- [ ] recon<br/>

## Features
### 1. Universal point set operator: intra-set, inter-set, and hier-set mixing <br/>
- Newly revisit the use of K-Nearest Neighbors <br/>
- Can process arbitrary number of points <br/>
<img src="./fig/universal point set operator.PNG" width="560" height="332"> <br/>

### 2. Symmetric encoder-decoder network for point clouds <br/>
- Maintain the hierarchical relation among points <br/>
- Design learning-based transition up/down layers (i.e., hier-set mixing) <br/>
<img src="./fig/symmetric.PNG" width="572" height="229"> <br/>

### 3. Parameter efficient design (**6.5M**) <br/>
<img src="./fig/arch.PNG" width="617" height="242"> <br/>


## References
```
@article{choe2021pointmixer,
  title={PointMixer: MLP-Mixer for Point Cloud Understanding},
  author={Choe, Jaesung and Park, Chunghyun and Rameau, Francois and Park, Jaesik and Kweon, In So},
  journal={arXiv preprint arXiv:2111.11187},
  year={2021}
}
```
