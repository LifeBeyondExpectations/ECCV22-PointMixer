#!/bin/sh

SERVER=${2:-local} 

if [[ $SERVER = *local* ]]; then
    echo "[PM INFO] Running on Local: You should manually load modules"
    conda init zsh
    source /opt/anaconda3/etc/profile.d/conda.sh
else
    echo "[PM INFO] Running on Server"
    conda init bash
    source ~/anaconda3/etc/profile.d/conda.sh

    module purge
    module load autotools 
    module load prun/1.3 
    module load gnu8/8.3.0 
    module load singularity
    
    module load cuDNN/cuda/11.1/8.0.4.30 
    module load cuda/11.1
    module load nccl/cuda/11.1/2.8.3

    echo "[PM INFO] Loaded all modules"
fi;

ENVS=$(conda env list | awk '{print $1}' )

if [[ $ENVS = *"$1"* ]]; then
    echo "[PM INFO] \"$1\" already exists. Pass the installation"
else
    echo "[PM INFO] Creating $1..."
    conda create -n $1 python=3.8 -y
    conda activate "$1"
    echo "[PM INFO] Done !"

    echo "[PM INFO] Installing PyTorch..."
    conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia -y
    echo "[PM INFO] Done !"

    echo "[PM INFO] Installing other dependencies..."
    conda install -c anaconda scipy h5py scikit-learn numpy -y
    conda install -c conda-forge einops cycler matplotlib pyyaml tqdm wandb -y
    pip install torch-scatter -f https://data.pyg.org/whl/torch-1.8.2+cu111.html
    echo "[PM INFO] Done !"

    echo "[PM INFO] Installing pointnet2_ops_lib..."
    pip install pointnet2_ops_lib/.
    echo "[PM INFO] Done !"

    echo "[PM INFO] Installing pointops2..."
    cd pointops2
    python3 setup.py install
    cd ..
    echo "[PM INFO] Done !"

    TORCH="$(python -c "import torch; print(torch.__version__)")"

    echo "[PM INFO] Finished the installation!"
    echo "[PM INFO] ========== Configurations =========="
    echo "[PM INFO] PyTorch version: $TORCH"
    echo "[PM INFO] ===================================="
fi;
