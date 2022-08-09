import os
import random
import pdb
import glob

import numpy as np
from plyfile import PlyData

import torch
import torch.utils.data

from dataset.utils.voxelize import voxelize
from utils.my_args import my_args

seed=0
# pl.seed_everything(seed) # , workers=True
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
    
def get_parser():
    parser = my_args()
    args = parser.parse_args()
    return args


def main():
    global args
    args = get_parser()
    test()




def data_load(data_name):
    if args.dataset == 'loader_s3dis':
        data_path = os.path.join(
            args.s3dis_root, 
            'trainval_fullarea',
             data_name+'.npy')
        data = np.load(data_path)  # xyzrgbl, N*7
        coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]

    elif args.dataset == 'loader_scannet':
        filepath = os.path.join(
            args.scannet_semgseg_root, 
            'val',
            data_name+'.pth')
        data = torch.load(filepath)
        coord, feat, label = data[0], data[1], data[2]

    elif args.dataset == 'loader_scannet_js':
        filepath = os.path.join(
            args.scannet_semgseg_root, 
            'train',
             data_name+'.ply')
        plydata = PlyData.read(filepath)
        data = plydata.elements[0].data
        coord = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
        feat = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
        label = np.array(data['label'], dtype=np.int32)

    else:
        raise NotImplemented

    idx_data = []
    if args.voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        idx_sort, count = voxelize(coord, args.voxel_size, mode=1)
        for i in range(count.max()):
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
            idx_part = idx_sort[idx_select]
            idx_data.append(idx_part)
    else:
        raise NotImplemented
        idx_data.append(np.arange(label.shape[0]))
    return coord, feat, label, idx_data


def test():

    if args.dataset == 'loader_s3dis':
        foldpath = os.path.join(args.s3dis_root, 'test_split')
        os.makedirs(foldpath, exist_ok=True)        
        # data_list = data_prepare()
        data_root = os.path.join(args.s3dis_root, 'trainval_fullarea')
        data_list = sorted(os.listdir(data_root))
        data_list = [item[:-4] for item in data_list if 
            'Area_{}'.format(args.test_area) in item]

    elif args.dataset == 'loader_scannet':
        # No GT label in test scenes.
        assert args.mode_eval == 'val' 
        foldpath = os.path.join(args.scannet_semgseg_root, 'val_split')
        os.makedirs(foldpath, exist_ok=True)
        data_root = os.path.join(args.scannet_semgseg_root, args.mode_eval)
        pth_list = sorted(glob.glob(os.path.join(data_root, '*.pth')))
        data_list = []
        for pth_path in pth_list:
            pth_file_name = pth_path.split('/')[-1] # 'scene0011_00_inst_nostuff.pth'
            pth_file_name = pth_file_name[:-4] # 'scene0011_00_inst_nostuff'
            data_list.append(pth_file_name)

    elif args.dataset == 'loader_scannet_js':
        # No GT label in test scenes.
        assert args.mode_eval == 'val' 
        foldpath = os.path.join(args.scannet_semgseg_root, 'val_split')
        os.makedirs(foldpath, exist_ok=True)
        
        # data_list = data_prepare()
        data_root = str(args.scannet_semgseg_root)
        scenes = read_txt(os.path.join(data_root, "scannetv2_val.txt"))
        
        # Be aware of "*.ply"
        # data_list = [os.path.join(data_root, 'train/%s.ply'%(scene)) for scene in scenes]
        # data_list = [os.path.join(data_root, 'train/%s'%(scene)) for scene in scenes]
        data_list = [scene for scene in scenes]

    else:
        raise NotImplemented
    print("Totally {} samples in val set.".format(len(data_list)))


    for idx, item in enumerate(data_list):

        coord, feat, label, idx_data = data_load(item)
        idx_size = len(idx_data)
        idx_list, coord_list, feat_list, offset_list, label_list = \
            [], [], [], [], []
        for i in range(idx_size):
            idx_part = idx_data[i]
            coord_part = coord[idx_part]
            feat_part = feat[idx_part]
            label_part = label[idx_part]

            if (args.eval_voxel_max) and (coord_part.shape[0] > args.eval_voxel_max):
                coord_p, idx_uni, cnt = \
                    np.random.rand(coord_part.shape[0]) * 1e-3, np.array([]), 0
                while idx_uni.size != idx_part.shape[0]:
                    init_idx = np.argmin(coord_p)
                    dist = np.sum(np.power(coord_part - coord_part[init_idx], 2), 1)
                    idx_crop = np.argsort(dist)[:args.eval_voxel_max]

                    coord_sub = coord_part[idx_crop]
                    feat_sub = feat_part[idx_crop]
                    idx_sub = idx_part[idx_crop]
                    label_sub = label_part[idx_crop]

                    dist = dist[idx_crop]
                    delta = np.square(1 - dist / np.max(dist))
                    coord_p[idx_crop] += delta
                    
                    # coord_sub, feat_sub = input_normalize(coord_sub, feat_sub)

                    idx_list.append(idx_sub)
                    coord_list.append(coord_sub)
                    feat_list.append(feat_sub)
                    offset_list.append(idx_sub.size)
                    label_list.append(label_sub)
                    idx_uni = np.unique(np.concatenate((idx_uni, idx_sub)))
            else:
                # coord_part, feat_part = input_normalize(coord_part, feat_part)

                idx_list.append(idx_part)
                coord_list.append(coord_part)
                feat_list.append(feat_part)
                offset_list.append(idx_part.size)
                label_list.append(label_part)
            
            # end of the for loop. (single batch per scene)
        
        for idx_batch in range(len(idx_list)):
            
            idx_part = idx_list[idx_batch]
            coord_part = coord_list[idx_batch]
            feat_part = feat_list[idx_batch]
            offset_part = offset_list[idx_batch]
            label_part = label_list[idx_batch]

            if args.dataset == 'loader_s3dis':
                filename = '{}__idx_{:05d}__npts_{:09d}.npz'.format(
                        item,
                        idx_batch,
                        args.eval_voxel_max)
            elif args.dataset == 'loader_scannet':
                filename = '{}__idx_{:05d}__npts_{:09d}__size0p{:04d}.npz'.format(
                        item, 
                        idx_batch, 
                        args.eval_voxel_max,
                        int(args.voxel_size*10000))
            else:
                raise NotImplemented
            filepath = os.path.join(foldpath, filename)
        

            print("Save files [%d/%d:%s] [%d/%d] coord_part.shape[0]=[%d] voxel_size[%.4f], eval_voxel_max[%d]"%(
                idx+1, len(data_list), filepath,
                idx_batch, len(idx_list), 
                coord_part.shape[0], 
                args.voxel_size,
                args.eval_voxel_max))

            np.savez(
                filepath, 
                idx_part=idx_part, 
                coord_part=coord_part, 
                feat_part=feat_part,
                offset_part=offset_part,
                label_part=label_part)

        
if __name__ == '__main__':
    main()
    
