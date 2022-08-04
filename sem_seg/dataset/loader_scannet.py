
from copy import deepcopy
import pdb
import os
import random
import glob

import numpy as np
import SharedArray as SA
from plyfile import PlyData

import torch
from torch.utils.data import Dataset

# from util.data_util import sa_create, collate_fn
from dataset.utils.data_util import data_prepare_scannet as data_prepare
from dataset.utils.data_util import sa_create
from dataset.utils import transform_scannet as transform
# from dataset.utils import transform as t
# from dataset.utils.voxelize import voxelize

seed=0
# pl.seed_everything(seed, workers=True)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if use multi-GPU
# torch.backends.cudnn.deterministic=True
# torch.backends.cudnn.benchmark=False

def TrainValCollateFn(batch):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    for item in coord:
        count += item.shape[0]
        offset.append(count)
    data_dict = \
        {
            'coord': torch.cat(coord),
            'feat': torch.cat(feat),
            'target': torch.cat(label),
            'offset': torch.IntTensor(offset),
        }
    return data_dict


def TestCollateFn(batch):
    coord, feat, label, pred_idx, offset = list(zip(*batch))
    data_dict = \
        {
            'coord': torch.cat(coord),
            'feat': torch.cat(feat),
            'target': torch.cat(label),
            'offset': torch.IntTensor(np.cumsum(offset)),
            'pred_idx': torch.cat(pred_idx),
        }
    return data_dict


class myImageFloder(Dataset):
    
    def __init__(self, args, mode, test_split=None):
        super().__init__()

        self.classes = float(args.classes) 
        assert self.classes == 20, 'this is scannet v2'
        self.ignore_label = int(args.ignore_label) # 255

        self.mode = mode # self.split = split
        data_root = deepcopy(args.scannet_semgseg_root) # self.data_root = data_root
        self.data_root = data_root
        self.voxel_size = float(args.voxel_size) # 0.04# self.voxel_size = voxel_size
        # self.voxel_max = voxel_max
        # self.transform = transform
        # self.shuffle_index = shuffle_index
        # self.loop = loop

        data_list = []
        if 'train' in mode:
            data_list += glob.glob(os.path.join(data_root, "train", "*.pth"))
        if 'val' in mode:
            data_list += glob.glob(os.path.join(data_root, "val", "*.pth"))
        if 'test' in mode:
            data_list += glob.glob(os.path.join(data_root, "test", "*.pth"))
            raise NotImplementedError
        assert len(data_list) > 0, f'len(data_list) = {len(data_list):d}'

        if mode == 'train' or mode == 'trainval':
            self.voxel_max = int(args.train_voxel_max)
            self.transform = transform.Compose([
                transform.RandomRotate(along_z=True),
                transform.RandomScale(scale_low=0.8, scale_high=1.2),
                transform.RandomDropColor(color_augment=0.0)])
            self.shuffle_index = True
            self.loop = int(args.loop)
            self.data_list = data_list

        elif mode == 'val':
            # self.data_list = glob.glob(os.path.join(data_root, mode, "*.pth"))
            self.voxel_max = int(args.eval_voxel_max) # 40000
            self.transform = None
            self.shuffle_index = False
            self.loop = 1
            
            self.test_split = test_split
            if self.test_split is None:
                self.data_list = data_list
            else:
                self.load_test_data()

        elif mode == 'test':
            raise NotImplementedError
        else:
            raise ValueError("no such mode: {}".format(mode))

    def __len__(self):
        return len(self.data_list) * self.loop

    def __getitem__(self, idx):
        
        if ((self.mode == 'train') 
            or (self.mode == 'trainval')
            or (self.mode == 'val' and self.test_split is None)):
            data_idx = idx % len(self.data_list)
            data_path = self.data_list[data_idx]
            data = torch.load(data_path)

            coord, feat, label = data[0], data[1], data[2]
            
            label[label == -100] = self.ignore_label
            
            coord, feat, label = data_prepare(
                coord, feat, label, 
                self.mode, self.voxel_size, self.voxel_max, 
                self.transform, self.shuffle_index)
            return coord, feat, label
        
        elif (self.mode == 'val' and self.test_split is not None):
            data_idx = self.data_idx[idx]
            data_path = self.data_list[data_idx]
            data = np.load(data_path) 

            pred_idx = data['idx_part']
            coord = data['coord_part']
            feat = data['feat_part'] # feat.max() == 255
            label = data['label_part']
            offset = data['offset_part']

            coord, feat, label = data_prepare(
                coord, feat, label, 
                split=None, voxel_size=None, voxel_max=None, 
                transform=None, shuffle_index=None)
            
            pred_idx = torch.LongTensor(pred_idx)
            label[label == -100] = self.ignore_label
            return coord, feat, label, pred_idx, offset
        
        elif mode == 'test':
            raise NotImplementedError
        else:
            raise ValueError("no such mode: {}".format(mode))
    
    def load_test_data(self):
        filename = self.test_split + '__' + '*__npts_{:09d}__size0p{:04d}.npz'.format(
            self.voxel_max, int(self.voxel_size*10000))
        self.data_list = sorted(glob.glob(os.path.join(self.data_root, 'val_split', filename)))
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), self.mode))
        
        # DO NOT ERASE 
        #
        # scenes = self.read_txt(os.path.join(self.data_root, "scannetv2_test.txt"))
        # self.data_paths = [os.path.join(self.data_root, 'test/%s.ply'%(scene)) for scene in scenes]
        # # self.data_paths = sorted(glob.glob(os.path.join(self.data_root, 'test/*.ply')))
        # print("Totally {} samples in {} set.".format(len(self.data_paths), self.mode))       



