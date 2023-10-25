
from copy import deepcopy
import pdb
import os
import random
import glob

import numpy as np
import SharedArray as SA
from colorama import Fore, Back, Style

import torch
from torch.utils.data import Dataset

from dataset.utils.data_util import data_prepare_v101 as data_prepare
from dataset.utils.data_util import sa_create
from dataset.utils import transform as t

seed=0
# pl.seed_everything(seed) # , workers=True
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

        self.mode = mode
        self.s3dis_root = deepcopy(args.s3dis_root)
        self.data_root = os.path.join(self.s3dis_root, 'trainval_fullarea')
        self.voxel_size = float(args.voxel_size) # 0.04
        self.test_area = int(args.test_area) # 5

        self.classes = float(args.classes) 
        assert self.classes == 13

        if self.mode == 'train':
            self.shuffle_index = True
            self.voxel_max = int(args.train_voxel_max)
            self.loop = int(args.loop)
            assert self.loop == 30
            self.transform = t.Compose(
                [
                    t.RandomScale([0.9, 1.1]), 
                    t.ChromaticAutoContrast(), 
                    t.ChromaticTranslation(), 
                    t.ChromaticJitter(), 
                    t.HueSaturationTranslation()
                ])
            self.load_trainval_data()

        elif self.mode == 'val':
            self.shuffle_index = False
            self.voxel_max = int(args.eval_voxel_max)
            self.loop = 1
            self.transform = None
            self.load_trainval_data()

        elif self.mode == 'test':
            self.shuffle_index = False
            self.voxel_max = int(args.eval_voxel_max)
            self.loop = 1
            self.transform = None

            self.test_split = test_split
            assert self.test_split is not None
            
            self.load_test_data()

        else:
            raise NotImplemented
    
    def __len__(self):
        return len(self.data_idx) * self.loop
    
    def load_trainval_data(self):
        data_list = sorted(os.listdir(self.data_root))
        data_list = [item[:-4] for item in data_list if 'Area_' in item]
        if self.mode == 'train':
            self.data_list = \
                [item for item in data_list if not 'Area_{}'.format(self.test_area) in item]
        elif self.mode == 'val':
            self.data_list = \
                [item for item in data_list if 'Area_{}'.format(self.test_area) in item]
        else:
            raise NotImplemented
        print(Fore.LIGHTYELLOW_EX + 'init Shared Array' + Style.RESET_ALL)
        for item in self.data_list:
            if not os.path.exists("/dev/shm/{}".format(item)):
                data_path = os.path.join(self.data_root, item + '.npy')
                data = np.load(data_path)  # xyzrgbl, N*7
                sa_create("shm://{}".format(item), data)
        self.data_idx = np.arange(len(self.data_list))
        print(Fore.LIGHTYELLOW_EX + 
            "Totally {} samples in {} set.".format(len(self.data_idx), self.mode)
             + Style.RESET_ALL)
    
    def load_test_data(self):
        filename = self.test_split + '__' + '*__npts_{:09d}.npz'.format(self.voxel_max)
        self.data_list = sorted(glob.glob(os.path.join(self.s3dis_root, 'test_split', filename)))
        self.data_idx = np.arange(len(self.data_list))
        print(Fore.LIGHTYELLOW_EX + 
            "Totally {} samples in {} set.".format(len(self.data_idx), self.mode) 
            + Style.RESET_ALL)

    def __getitem__(self, idx):
        if (self.mode == 'train') or (self.mode == 'val'):
            data_idx = self.data_idx[idx % len(self.data_idx)]
            data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
            coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
            # feat.max() == 255.
            coord, feat, label = data_prepare(
                coord, feat, label, 
                self.mode, self.voxel_size, self.voxel_max, 
                self.transform, self.shuffle_index)
            return coord, feat, label

        elif (self.mode == 'test'):
            data_idx = self.data_idx[idx]
            data_path = self.data_list[data_idx]
            data = np.load(data_path) 

            pred_idx = data['idx_part']
            coord = data['coord_part']
            feat = data['feat_part']
            label = data['label_part']
            offset = data['offset_part']

            coord, feat, label = data_prepare(
                coord, feat, label, 
                split=None, voxel_size=None, voxel_max=None, 
                transform=None, shuffle_index=None)

            pred_idx = torch.LongTensor(pred_idx)

            return coord, feat, label, pred_idx, offset

        else:
            raise NotImplemented
