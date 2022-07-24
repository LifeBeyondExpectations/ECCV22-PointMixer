FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
MAINTAINER jaesung.choe@kaist.ac.kr

ENV PATH /opt/conda/bin:$PATH

RUN apt-get -y update
RUN apt-get install -y git vim screen python3-dev python3-pip wget curl libgl1-mesa-glx rsync
RUN apt install -y build-essential python3-dev libopenblas-dev
RUN echo "export PATH=\$PATH:/usr/local/cuda/bin" >> /root/.bashrc
RUN echo "export CUDADIR=/usr/local/cuda" >> /root/.bashrc
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64" >> /root/.bashrc
RUN echo "export CUDA_HOME=/usr/local/cuda" >> /root/.bashrc

RUN wget --quiet http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda.sh && \
    /bin/bash /root/miniconda.sh -b -p /opt/conda && \
    rm /root/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda activate base" >> /root/.bashrc

ENV PATH=/miniconda/bin:$PATH

### torch install
# RUN apt-get update -y
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

### Dependency
RUN conda install -c anaconda h5py pyyaml scikit-learn python-lmdb -y
RUN conda install -c conda-forge sharedarray tensorboardx einops tqdm msgpack-numpy -y
# RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
RUN pip3 install opencv-python
RUN pip3 install matplotlib==3.3.4
# RUN pip3 install tensorboardX==2.1
RUN pip3 install scikit-image==0.17.2
RUN pip3 install pytorch-lightning==1.5.7
RUN pip3 install neptune-client==0.14.2
RUN pip3 install open3d
RUN pip3 install pykitti==0.3.1
RUN pip3 install torchmetrics
# RUN pip3 install einops
RUN pip3 install pandas==1.3.0
RUN pip3 install plyfile
RUN pip3 install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
RUN pip3 install setuptools==59.5.0

### Oh My Zsh install for Docker
# https://github.com/deluan/zsh-in-docker
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.2/zsh-in-docker.sh)" -- \
-a 'CASE_SENSITIVE="true"'