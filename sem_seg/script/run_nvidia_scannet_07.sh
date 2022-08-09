#!/bin/bash

### Path
PU_GAN_MESH=/root/dataset/PU-GAN
SCANNET_TRAIN=/root/dataset/deepmvs/train
SCANNET_TEST=/root/dataset/deepmvs/test
SCANNET_SEMSEG=/root/dataset/scannet_semseg
SHAPENET=/root/dataset/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/
SHAPNETCORE=/root/dataset/shapenetcore/ShapeNetCore.v2/
S3DIS=/root/dataset/S3DIS/s3dis/
SAVEROOT="/root/PointMixerSemSeg/"

### Setup 
MYSHELL="run_nvidia_scannet_07.sh"
DATE_TIME=`date +"%Y-%m-%d"`
NEPTUNE_PROJ="jaesung.choe/ECCV22-PointMixer-SemSeg"
COMPUTER="SCANNET-NVIDIA-07-PointGroup"
export MASTER_ADDR='localhost'
export NODE_RANK=0
export CUDA_VISIBLE_DEVICES=0

### Params
WORKERS=4
NUM_GPUS=1
NUM_TRAIN_BATCH=4
NUM_VAL_BATCH=2
NUM_TEST_BATCH=4
VOX_SIZE=0.05
LOOP=10
N_PTS=50000

ARCH="pointmixer"
DATASET="loader_scannet" # "loader_s3dis"
INTRALAYER="PointMixerIntraSetLayer"
INTERLAYER="PointMixerInterSetLayer"
TRANSDOWN="SymmetricTransitionDownBlock"
TRANSUP="SymmetricTransitionUpBlock"

MYCHECKPOINT="${SAVEROOT}/${DATE_TIME}__${DATASET}__\
${INTRALAYER}__${INTERLAYER}__${TRANSDOWN}__${TRANSUP}__${COMPUTER}/"
reset
rm -rf $MYCHECKPOINT
mkdir -p $MYCHECKPOINT
# cp -a "../../sem_seg" $MYCHECKPOINT
cp -a $MYSHELL $MYCHECKPOINT
cd ../
sh env_setup.sh

### TRAIN
python train_pl.py \
  --MYCHECKPOINT $MYCHECKPOINT --computer $COMPUTER --shell $MYSHELL \
  --MASTER_ADDR $MASTER_ADDR \
  --train_worker $WORKERS --val_worker $WORKERS \
  --NUM_GPUS $NUM_GPUS  \
  --train_batch $NUM_TRAIN_BATCH  \
  --val_batch $NUM_VAL_BATCH  \
  --test_batch $NUM_TEST_BATCH \
  \
  --scannet_train_root $SCANNET_TRAIN  --scannet_test_root $SCANNET_TEST \
  --scannet_semgseg_root $SCANNET_SEMSEG \
  --shapenet_root $SHAPENET  --shapenetcore_root $SHAPNETCORE \
  --s3dis_root $S3DIS \
  \
  --neptune_proj $NEPTUNE_PROJ \
  --epochs 60  --CHECKPOINT_PERIOD 1  --lr 0.1 \
  --dataset $DATASET  --distributed_backend 'dp' --optim 'SGD' \
  \
  --model 'net_pointmixer' --arch $ARCH  \
  --intraLayer $INTRALAYER  --interLayer $INTERLAYER \
  --transdown  $TRANSDOWN --transup $TRANSUP \
  --nsample 8 16 16 16 16  --drop_rate 0.1  --fea_dim 6  --classes 20  --loop $LOOP \
  \
  --test_batch 1  --cudnn_benchmark False \
  --voxel_size $VOX_SIZE  --train_voxel_max $N_PTS  --eval_voxel_max $N_PTS  \
  --mode_train 'train' --mode_eval 'val'  --aug 'elastic+mink'

### TEST (pre-process stage for test dataset)
python test_split_save.py \
  --MYCHECKPOINT $MYCHECKPOINT --computer $COMPUTER --shell $MYSHELL \
  --MASTER_ADDR $MASTER_ADDR \
  --train_worker $WORKERS --val_worker $WORKERS \
  --NUM_GPUS $NUM_GPUS  \
  --train_batch $NUM_TRAIN_BATCH  \
  --val_batch $NUM_VAL_BATCH  \
  --test_batch $NUM_TEST_BATCH \
  \
  --scannet_train_root $SCANNET_TRAIN  --scannet_test_root $SCANNET_TEST \
  --scannet_semgseg_root $SCANNET_SEMSEG \
  --shapenet_root $SHAPENET  --shapenetcore_root $SHAPNETCORE \
  --s3dis_root $S3DIS \
  \
  --neptune_proj $NEPTUNE_PROJ \
  --epochs 60  --CHECKPOINT_PERIOD 1  --lr 0.1 \
  --dataset $DATASET  --distributed_backend 'dp' --optim 'SGD' \
  \
  --model 'net_pointmixer' --arch $ARCH  \
  --intraLayer $INTRALAYER  --interLayer $INTERLAYER \
  --transdown  $TRANSDOWN --transup $TRANSUP \
  --nsample 8 16 16 16 16  --drop_rate 0.1  --fea_dim 6  --classes 20  --loop $LOOP \
  \
  --test_batch 1  --cudnn_benchmark False \
  --voxel_size $VOX_SIZE  --train_voxel_max $N_PTS  --eval_voxel_max $N_PTS  \
  --mode_train 'train' --mode_eval 'val'  --aug 'elastic+mink'

### TEST (evaluation)
python test_pl.py \
  --MYCHECKPOINT $MYCHECKPOINT --computer $COMPUTER --shell $MYSHELL \
  --MASTER_ADDR $MASTER_ADDR \
  --train_worker $WORKERS --val_worker $WORKERS \
  --NUM_GPUS $NUM_GPUS  \
  --train_batch $NUM_TRAIN_BATCH  \
  --val_batch $NUM_VAL_BATCH  \
  --test_batch $NUM_TEST_BATCH \
  \
  --scannet_train_root $SCANNET_TRAIN  --scannet_test_root $SCANNET_TEST \
  --scannet_semgseg_root $SCANNET_SEMSEG \
  --shapenet_root $SHAPENET  --shapenetcore_root $SHAPNETCORE \
  --s3dis_root $S3DIS \
  \
  --neptune_proj $NEPTUNE_PROJ \
  --epochs 60  --CHECKPOINT_PERIOD 1  --lr 0.1 \
  --dataset $DATASET  --distributed_backend 'dp' --optim 'SGD' \
  \
  --model 'net_pointmixer' --arch $ARCH  \
  --intraLayer $INTRALAYER  --interLayer $INTERLAYER \
  --transdown  $TRANSDOWN --transup $TRANSUP \
  --nsample 8 16 16 16 16  --drop_rate 0.1  --fea_dim 6  --classes 20  --loop $LOOP \
  \
  --test_batch 1  --cudnn_benchmark False \
  --voxel_size $VOX_SIZE  --train_voxel_max $N_PTS  --eval_voxel_max $N_PTS  \
  --mode_train 'train' --mode_eval 'val'  --aug 'elastic+mink'

  cd -