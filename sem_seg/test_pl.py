from __future__ import print_function
import random
import shutil
import os
import glob
from copy import deepcopy

import torch # why is it located here?
import numpy as np
from plyfile import PlyData
import pdb
import cv2
cv2.setNumThreads(0)
import pytorch_lightning as pl

from utils.my_args import my_args
from utils.common_util import AverageMeter, intersectionAndUnion, find_free_port
from model import get as get_model
from dataset import get as get_dataset

seed=0
pl.seed_everything(seed) # , workers=True
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if use multi-GPU
# torch.backends.cudnn.deterministic=True
# torch.backends.cudnn.benchmark=False

def read_txt(path):
    with open(path) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines

# https://github.com/PyTorchLightning/deep-learning-project-template/blob/master/project/lit_mnist.py
def cli_main():

    # ------------
    # args
    # ------------
    parser = my_args()
    args = parser.parse_args()


    # ------------
    # randomness or seed
    # ------------
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    free_port = find_free_port()
    os.environ["MASTER_PORT"] = str(free_port)


    # ------------
    # logger
    # ------------
    from pytorch_lightning.loggers import NeptuneLogger
    neptune_path = os.path.join(args.MYCHECKPOINT, 'neptune.npz')
    if args.neptune_id:
        np.savez(
            neptune_path,
            project=args.neptune_proj,
            id=args.neptune_id)
        print(" >> newly create naptune.npz in test_pl.py")

    if os.path.exists(neptune_path):
        neptune_info = np.load(neptune_path)
        
        neptune_logger = NeptuneLogger(
            api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4MmJmODE2Ni1jZDE0LTRjY2MtYjViYS0wMjAwNWYzOWQzMjIifQ==", 
            project=args.neptune_proj)
        neptune_logger._run_short_id = str(neptune_info['id'])
        print(">> re-use the neptune: id[%s]"%(neptune_info['id']))
    else:
        neptune_logger = NeptuneLogger(
            api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4MmJmODE2Ni1jZDE0LTRjY2MtYjViYS0wMjAwNWYzOWQzMjIifQ==", 
            project=args.neptune_proj)
        print(">> start new neptune")

    neptune_logger.experiment["sys/tags"].add('test_pl.py')

    # # only update in train_pl.py
    # for key, value in (vars(args)).items():
    #     neptune_logger.experiment['params/' + key] = value


    # ------------
    # model
    # ------------
    ckpts = sorted(glob.glob(os.path.join(args.MYCHECKPOINT, "*.ckpt")))
    if len(ckpts)>1: ckpts = ckpts[:-1] # remove 'last.ckpt'
    mIoU_val_best = -1.
    filename_best = None
    for ckpt in ckpts:
        rootpath = '/'.join(ckpt.split('/')[:-1])
        filename = ckpt.split('/')[-1] # 'epoch=041--mIoU_val=0.6800--.ckpt'
        mIoU_val = filename[:-5] # 'epoch=041--mIoU_val=0.6800--'
        mIoU_val = float((mIoU_val.split('--')[-2])[9:])
        if mIoU_val >= mIoU_val_best:
            mIoU_val_best = mIoU_val
            filename_best = filename
    ckpt_best = os.path.join(rootpath, filename_best)
    args.load_model = ckpt_best
    args.on_train = False
    print('ckpt best. args.load_model=[{}]'.format(args.load_model))
    assert args.load_model is not None, 'why did you come?'
    model = get_model(args.model).load_from_checkpoint(
        os.path.join(args.MYCHECKPOINT, args.load_model), 
        args=args, 
        strict=True) # args.strict_load

    model.eval()
    model.freeze()

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer(
        logger=neptune_logger,
        gpus=1,
        enable_progress_bar=False if 'nvidia' in args.computer else True,
    )

    # ------------
    # test
    # ------------
    if args.dataset == 'loader_s3dis':
        data_root = os.path.join(args.s3dis_root, 'trainval_fullarea')
        data_list = sorted(os.listdir(data_root))
        data_list = [item[:-4] for item in data_list if 'Area_{}'.format(args.test_area) in item]
    elif args.dataset == 'loader_scannet':
        # No GT label in test scenes.
        # assert args.mode_eval == 'test' 
        foldpath = os.path.join(args.scannet_semgseg_root, 'val_split')
        os.makedirs(foldpath, exist_ok=True)
        data_root = str(args.scannet_semgseg_root)
        scenes = read_txt(os.path.join(data_root, "scannetv2_val.txt"))
        data_list = [scene for scene in scenes]
    else:
        raise NotImplemented

    ckpt_name = (args.load_model).split('/')[-1]
    ckpt_name = ckpt_name[:-5] # remove '.ckpt'
    save_folder = os.path.join(args.MYCHECKPOINT, 'test_results__%s'%((ckpt_name)))
    os.makedirs(save_folder, exist_ok=True)

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    pred_save, label_save = [], []

    str_to_log = '<<<<<<<<<<<<<<<<< Start Evaluation <<<<<<<<<<<<<<<<<'
    print(str_to_log)
    neptune_logger.experiment['logs'].log(str_to_log)
    
    for idx, item in enumerate(data_list):
        
        pred_save_filename = \
            '{}__epoch_{}npts{:09d}__size0p{:04d}__pred__test_pl__.npy'.format(
                item, model.current_epoch, args.eval_voxel_max, int(args.voxel_size*10000))
        pred_save_path = os.path.join(save_folder, pred_save_filename)

        label_save_filename = \
            '{}__epoch_{}npts{:09d}__size0p{:04d}__label__test_pl__.npy'.format(
                item, model.current_epoch, args.eval_voxel_max, int(args.voxel_size*10000))
        label_save_path = os.path.join(save_folder, label_save_filename)


        if os.path.isfile(pred_save_path) and os.path.isfile(label_save_path):
            print('{}/{}: [{}], loaded pred and label.'.format(
                idx+1, len(data_list), item))
            pred = np.load(pred_save_path)
            label = np.load(label_save_path)
            
        else:

            if args.dataset == 'loader_s3dis':
                data_path = os.path.join(
                    args.s3dis_root, 'trainval_fullarea', item+'.npy')
                data = np.load(data_path)
                label = data[:, 6] # coord, feat = data[:, :3], data[:, 3:6]
            elif args.dataset == 'loader_scannet':
                filepath = os.path.join(
                    args.scannet_semgseg_root, 'train/%s.ply'%(item))
                plydata = PlyData.read(filepath)
                data = plydata.elements[0].data
                label = np.array(data['label'], dtype=np.int32)
            else:
                raise NotImplemented
            


            with torch.no_grad():
                model.pred = torch.zeros((label.size, args.classes)).cuda()

            dataset = get_dataset(args.dataset)
            test_loader_kwargs = \
                {
                    "batch_size": args.test_batch, # WRONG. Because of my stupid code. ,
                    "num_workers": args.val_worker,
                    "collate_fn": dataset.TestCollateFn,
                    "pin_memory": False,
                    "drop_last": False,
                    "shuffle": False,
                }

            test_loader = torch.utils.data.DataLoader(
                dataset.myImageFloder(args, mode='test', test_split=item), **test_loader_kwargs)

            trainer.test(model=model, dataloaders=test_loader, verbose=True)

            pred = model.pred.max(1)[1].cpu().detach().numpy()

            np.save(pred_save_path, pred)
            np.save(label_save_path, label)
            # end of it cond

        # calculation 1: add per room predictions
        intersection, union, target = \
            intersectionAndUnion(pred, label, args.classes, args.ignore_label)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        
        accuracy = sum(intersection) / (sum(target) + 1e-10)
        neptune_logger.experiment['acc_test_per_scene'].log(accuracy)

        str_to_log = \
            'Test: [{:4d}/{:4d}]-npts[{npts:7d}/{:7d}] Accuracy[{accuracy:.4f}]'.format(
                int(idx+1), len(data_list), int(label.size),
                accuracy=accuracy, 
                npts=args.eval_voxel_max)
        print(str_to_log)
        neptune_logger.experiment['logs'].log(str_to_log)

        pred_save.append(pred)
        label_save.append(label)
        # end of the for loop. (per-scene)

    # calculation 1
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU1 = np.mean(iou_class)
    mAcc1 = np.mean(accuracy_class)
    allAcc1 = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    # calculation 2
    intersection, union, target = \
        intersectionAndUnion(
            np.concatenate(pred_save), 
            np.concatenate(label_save), 
            args.classes, 
            args.ignore_label)
    iou_class = intersection / (union + 1e-10)
    accuracy_class = intersection / (target + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection) / (sum(target) + 1e-10)

    str_to_log = \
        'Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(
            mIoU, mAcc, allAcc)
    print(str_to_log)
    neptune_logger.experiment['logs'].log(str_to_log)

    str_to_log = \
        'Val1 result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(
            mIoU1, mAcc1, allAcc1)
    print(str_to_log)
    neptune_logger.experiment['logs'].log(str_to_log)

    neptune_logger.experiment['mIoU_test'].log(mIoU)
    neptune_logger.experiment['mAcc_test'].log(mAcc)
    neptune_logger.experiment['allAcc_test'].log(allAcc)
    neptune_logger.experiment['mIoU_test1'].log(mIoU1)
    neptune_logger.experiment['mAcc_test1'].log(mAcc1)
    neptune_logger.experiment['allAcc_test1'].log(allAcc1)
    
    if args.dataset == 'loader_s3dis':
        names_path = os.path.join(args.s3dis_root, "list/s3dis_names.txt")
        names = [line.rstrip('\n') for line in open(names_path)]
    elif args.dataset == 'loader_scannet':
        names = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
    else:
        print("Please set labels' names.")
        raise NotImplemented

    for i in range(args.classes):
        str_to_log = \
            'Class_{} Result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(
                i, iou_class[i], accuracy_class[i], names[i])
        print(str_to_log)
        neptune_logger.experiment['logs'].log(str_to_log)

        neptune_logger.experiment['class_acc_test_%d_%s'%(i, names[i])].log(accuracy_class[i])
        neptune_logger.experiment['class_iou_test_%d_%s'%(i, names[i])].log(iou_class[i])

    str_to_log = '<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<'
    print(str_to_log)
    neptune_logger.experiment['logs'].log(str_to_log)
    
    neptune_logger.experiment["sys/failed"] = False
    neptune_logger.experiment.stop()

if __name__ == '__main__':
    cli_main()