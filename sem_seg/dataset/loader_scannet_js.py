
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
from dataset.utils.data_util import data_prepare_v101 as data_prepare
from dataset.utils.data_util import sa_create
from dataset.utils import transform as t
# from dataset.utils.voxelize import voxelize

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
        self.data_root = deepcopy(args.scannet_semgseg_root)
        self.voxel_size = float(args.voxel_size) # 0.04

        self.classes = float(args.classes) 
        assert self.classes == 20        

        self.aug = str(args.aug)
        self.crop_npart = int(args.crop_npart)

        if self.mode == 'train':
            self.shuffle_index = True
            self.voxel_max = int(args.train_voxel_max)
            self.loop = int(args.loop)

            if self.aug == 'pointtransformer':
                self.transform = t.Compose(
                    [
                        t.RandomScale([0.9, 1.1]), 
                        t.ChromaticAutoContrast(), 
                        t.ChromaticTranslation(), 
                        t.ChromaticJitter(), 
                        t.HueSaturationTranslation()
                    ])
            elif self.aug == 'pointtransformer-v2':
                self.transform = t.Compose(
                    [
                        t.ChromaticAutoContrast(), 
                        t.ChromaticTranslation(), 
                        t.ChromaticJitter(), 
                        t.HueSaturationTranslation()
                    ])
            elif self.aug == 'mink':
                self.transform = t.Compose(
                    [
                        t.RandomHorizontalFlip('z'),
                        t.ChromaticAutoContrast(),
                        t.ChromaticTranslation(ratio=0.10),
                        t.ChromaticJitter(std=0.05),
                    ])
            elif self.aug == 'elastic+mink':
                self.transform = t.Compose(
                    [
                        t.ElasticDistortion(),
                        t.RandomHorizontalFlip('z'),
                        t.ChromaticAutoContrast(),
                        t.ChromaticTranslation(ratio=0.10),
                        t.ChromaticJitter(std=0.05),
                    ])
            elif self.aug == 'elastic+mink+crop':
                self.transform = t.Compose(
                    [
                        t.ElasticDistortion(),
                        t.CoordCrop(npart=self.crop_npart), # jaesungchoe
                        t.RandomHorizontalFlip('z'),
                        t.ChromaticAutoContrast(),
                        t.ChromaticTranslation(ratio=0.10),
                        t.ChromaticJitter(std=0.05),
                    ])
            elif self.aug == 'mink+crop':
                self.transform = t.Compose(
                    [
                        t.CoordCrop(npart=self.crop_npart), # jaesungchoe
                        t.RandomHorizontalFlip('z'),
                        t.ChromaticAutoContrast(),
                        t.ChromaticTranslation(ratio=0.10),
                        t.ChromaticJitter(std=0.05),
                    ])
            else:
                raise NotImplemented
            self.load_train_data()
        
            # elif self.mode == 'trainval':
            #     self.shuffle_index = True
            #     self.voxel_max = int(args.train_voxel_max) # 40000
            #     self.loop = int(args.loop)
            #     self.transform = t.Compose(
            #         [
            #             t.RandomScale([0.9, 1.1]), 
            #             t.ChromaticAutoContrast(), 
            #             t.ChromaticTranslation(), 
            #             t.ChromaticJitter(), 
            #             t.HueSaturationTranslation()
            #         ])
            #     self.load_trainval_data()

        elif self.mode == 'val':
            self.shuffle_index = False
            self.voxel_max = int(args.eval_voxel_max) # 800000
            self.loop = 1
            self.transform = None
                
            self.load_val_data()

        elif self.mode == 'test':
            self.shuffle_index = False
            self.voxel_max = int(args.eval_voxel_max) # 40000
            self.loop = 1
            self.transform = None

            self.test_split = test_split
            assert self.test_split is not None
            
            self.load_test_data()

        else:
            raise NotImplemented

        ### DEBUG
        # for idx in range(self.__len__()):
        #     print(f'idx[{idx:4d}]')
        #     self.__getitem__(idx=idx)

    def __getitem__(self, idx):

        if self.mode in ['train', 'val', 'trainval']:
            idx = idx % len(self.data_paths)
            filepath = self.data_paths[idx]
            plydata = PlyData.read(filepath)
            data = plydata.elements[0].data
            coord = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
            feat = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
            label = np.array(data['label'], dtype=np.int32)
            # feat.max() == 255            
            # print("mode[%s], idx[%d], coord[%d]"%(self.mode, idx, coord.shape[0])); return
            coord, feat, label = data_prepare(
                coord, feat, label, 
                self.mode, self.voxel_size, self.voxel_max, 
                self.transform, self.shuffle_index)
            # feat.max() == 1.0

            # pdb.set_trace()
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(coord)
            # pcd.colors = o3d.utility.Vector3dVector(feat)
            # o3d.io.write_point_cloud('/root/scannet_sampled_input.ply', pcd)

            return coord, feat, label

        elif (self.mode == 'test'):
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

            return coord, feat, label, pred_idx, offset

        else:
            raise NotImplemented

    def load_train_data(self):
        scenes = self.read_txt(os.path.join(self.data_root, "scannetv2_train.txt"))        
        self.data_paths = [os.path.join(self.data_root, 'train/%s.ply'%(scene)) for scene in scenes]
        print("Totally {} samples in {} set.".format(len(self.data_paths), self.mode))
    
    def load_val_data(self):
        scenes = self.read_txt(os.path.join(self.data_root, "scannetv2_val.txt"))
        self.data_paths = [os.path.join(self.data_root, 'train/%s.ply'%(scene)) for scene in scenes]
        print("Totally {} samples in {} set.".format(len(self.data_paths), self.mode))

    def load_trainval_data(self):
        scenes_train = self.read_txt(os.path.join(self.data_root, "scannetv2_train.txt"))        
        scenes_val = self.read_txt(os.path.join(self.data_root, "scannetv2_val.txt"))
        scenes = scenes_train + scenes_val
        self.data_paths = [os.path.join(self.data_root, 'train/%s.ply'%(scene)) for scene in scenes]
        print("Totally {} samples in {} set.".format(len(self.data_paths), self.mode))
    
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

    def __len__(self):
        if self.mode in ['train', 'val', 'trainval']:
            return len(self.data_paths) * self.loop
        elif (self.mode == 'test'):
            return len(self.data_idx) * self.loop
        else:
            raise NotImplemented
    
    def read_txt(self, path):
        with open(path) as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]
        return lines