import argparse

def str2bool(x):
    return x.lower() in ('true')

def my_args():
    parser = argparse.ArgumentParser()


    # ------------
    # model
    # ------------
    parser.add_argument('--model', default='stackhourglass', help='select model')


    # ------------
    # network hyper-parms
    # ------------
    parser.add_argument('--arch', default='pointmixer')
    parser.add_argument('--intraLayer', default='intraLayer')
    parser.add_argument('--interLayer', default='interLayer')
    parser.add_argument('--transdown', default='SymmetricTransitionDownBlockPaperv3')
    parser.add_argument('--transup', default='SymmetricTransitionUpBlock')
    parser.add_argument("--nsample", nargs="+", type=int, default=[8, 16, 16, 16, 16])
    parser.add_argument("--downsample", nargs="+", type=int, default=[1, 4, 4, 4, 4])
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--fea_dim", type=int, default=6, help='input point feat dim') 
    parser.add_argument("--classes", type=int, default=13, help='output classes')


    # ------------
    # randomness or seed
    # ------------
    parser.add_argument('--cudnn_benchmark', type=str2bool, default=False)


    # ------------
    # Dataloader
    # ------------
    parser.add_argument('--train_batch', type=int, default=4)
    parser.add_argument('--val_batch', type=int, default=4)
    parser.add_argument('--test_batch', type=int, default=10)
    parser.add_argument('--train_worker', type=int, default=4)
    parser.add_argument('--val_worker', type=int, default=4)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument('--scannet_train_root')
    parser.add_argument('--scannet_test_root')
    parser.add_argument('--scannet_semgseg_root')
    parser.add_argument('--shapenet_root')
    parser.add_argument('--shapenetcore_root')
    parser.add_argument('--s3dis_root')
    
    parser.add_argument("--loop", type=int, default=30)
    parser.add_argument("--ignore_label", type=int, default=255) 
    parser.add_argument("--test_area", type=int, default=5) 

    parser.add_argument("--train_voxel_max", type=int, default=40000) 
    parser.add_argument("--eval_voxel_max", type=int, default=800000) 
    
    parser.add_argument("--voxel_size", type=float, default=0.04)

    parser.add_argument("--mode_train", type=str, default='train')
    parser.add_argument("--mode_eval", type=str, default='val')

    parser.add_argument("--aug", type=str, default='pointtransformer')
    parser.add_argument("--crop_npart", type=int, default=0) 


    # ------------
    # optimizer
    # ------------
    parser.add_argument("--optim", type=str, default='Adam')
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    
    parser.add_argument("--lr_STEP_SIZE", type=int, default=6)
    parser.add_argument("--lr_GAMMA", type=float, default=0.1)

    parser.add_argument("--schedule", nargs="+", type=float, default=[0.6, 0.8])


    # ------------
    # Trainer
    # ------------
    parser.add_argument('--AMP_LEVEL', type=str, default='O0') # 'O0' == 32bit, # '01' == 16 bit
    parser.add_argument('--PRECISION', type=int, default=32)
    parser.add_argument('--distributed_backend', type=str, default='ddp') # 'O0' == 32bit
    parser.add_argument('--load_model', default=None)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--on_train', type=str2bool, default=True)
    parser.add_argument('--shell', type=str, default='run_old.sh')
    parser.add_argument('--MYCHECKPOINT', default='./', help='save model')
    parser.add_argument("--computer", type=str, default=None)
    parser.add_argument('--NUM_GPUS', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--CHECKPOINT_PERIOD', type=int, default=1) 
    parser.add_argument('--strict_load', type=str2bool, default=True)
    
    parser.add_argument('--MASTER_ADDR', type=str, default='localhost')
    parser.add_argument('--MASTER_PORT', type=str, default='29500')
    

    # ------------
    # logger
    # ------------
    parser.add_argument('--on_neptune', action='store_true')
    parser.add_argument('--off_text_logger', action='store_true')
    parser.add_argument('--neptune_proj', type=str, default=None, required=True)
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--neptune_id', type=str, default=None)
    parser.add_argument('--neptune_key', type=str, default=None)

    return parser