import math
import pdb
import random

import numpy as np
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch_scatter import scatter, scatter_softmax, scatter_sum, scatter_std, scatter_max

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.pointops2.functions import pointops

seed=0
# pl.seed_everything(seed) # , workers=True
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if use multi-GPU
# torch.backends.cudnn.deterministic=True
# torch.backends.cudnn.benchmark=False

####################################################################################

class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes, nsample):
        super().__init__()
        self.nsample = nsample
        if out_planes is None:
            self.linear1 = nn.Sequential(
                nn.Linear(2*in_planes, in_planes), 
                nn.BatchNorm1d(in_planes), 
                nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, in_planes), 
                nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(
                nn.Linear(out_planes, out_planes), 
                nn.BatchNorm1d(out_planes), 
                nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, out_planes), 
                nn.BatchNorm1d(out_planes), 
                nn.ReLU(inplace=True))
        
    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i-1], o[i], o[i] - o[i-1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1; p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x

# SymmetricTransitionUpBlock
class SymmetricTransitionUpBlock(nn.Module):
    def __init__(self, in_planes, out_planes, nsample):
        super().__init__()
        self.nsample = nsample
        if out_planes is None:
            self.linear1 = nn.Sequential(
                nn.Linear(2*in_planes, in_planes, bias=False), 
                nn.BatchNorm1d(in_planes), 
                nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, in_planes), 
                nn.ReLU(inplace=True))            
        else:
            self.linear1 = nn.Sequential( # input.shape = [N, L]
                nn.Linear(out_planes, out_planes, bias=False), 
                nn.BatchNorm1d(out_planes),  
                nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential( # input.shape = [N, L]
                nn.Linear(in_planes, out_planes, bias=False), 
                nn.BatchNorm1d(out_planes), 
                nn.ReLU(inplace=True))
            self.channel_shrinker = nn.Sequential( # input.shape = [N*K, L]
                nn.Linear(in_planes+3, in_planes, bias=False),
                nn.BatchNorm1d(in_planes),
                nn.ReLU(inplace=True),
                nn.Linear(in_planes, 1))
        
    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i-1], o[i], o[i] - o[i-1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            y = self.linear1(x) # this part is the same as TransitionUp module.
        else:
            # x1.shape: (n, c) encoder/fine-grain points
            # x2.shape: (m, c) decoder/coase points
            # p1.shape: (436, 3) # (n, 3) # p1 is the upsampled one.
            # p2.shape: (109, 3) # (m, 3)
            p1, x1, o1 = pxo1; p2, x2, o2 = pxo2 
            knn_idx = pointops.knnquery(self.nsample, p1, p2, o1, o2)[0].long()
            # knn_idx.shape: (109, 16) # (m, nsample) 
            # knn_idx.max() == 435 == n # filled with x1's idx

            with torch.no_grad():
                knn_idx_flatten = rearrange(knn_idx, 'm k -> (m k)')
            p_r = p1[knn_idx_flatten, :].view(len(p2), self.nsample, 3) - p2.unsqueeze(1)
            x2_knn = x2.view(len(p2), 1, -1).repeat(1, self.nsample, 1)
            x2_knn = torch.cat([p_r, x2_knn], dim=-1) # (109, 16, 259) # (m, nsample, 3+c)

            with torch.no_grad():
                knn_idx_flatten = knn_idx_flatten.unsqueeze(-1) # (m*nsample, 1)
            x2_knn_flatten = rearrange(x2_knn, 'm k c -> (m k) c') # c = 3+out_planes
            x2_knn_flatten_shrink = self.channel_shrinker(x2_knn_flatten) # (m, nsample, 1)
            x2_knn_prob_flatten_shrink = scatter_softmax(
                x2_knn_flatten_shrink, knn_idx_flatten, dim=0)

            x2_knn_prob_shrink = rearrange(
                x2_knn_prob_flatten_shrink, '(m k) 1 -> m k 1', k=self.nsample)
            up_x2_weighted = self.linear2(x2).unsqueeze(1) * x2_knn_prob_shrink
            up_x2_weighted_flatten = rearrange(up_x2_weighted, 'm k c -> (m k) c')
            up_x2 = scatter_sum(
                up_x2_weighted_flatten, knn_idx_flatten, dim=0, dim_size=len(p1))
            y = self.linear1(x1) + up_x2
        return y

####################################################################################

class TransitionDownBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3+in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i-1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, 
                use_xyz=True)  # (m, nsample, 3+c)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, nsample, c)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]

# SymmetricTransitionDownBlock_ECCV22
class SymmetricTransitionDownBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, nsample):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        
        if stride != 1:
            self.linear2 = nn.Sequential( # input.shape = [N, L]
                nn.Linear(in_planes, out_planes, bias=False),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True))

            self.channel_shrinker = nn.Sequential( # input.shape = [N*K, L]
                nn.Linear(3+in_planes, in_planes, bias=False),
                nn.BatchNorm1d(in_planes),
                nn.ReLU(inplace=True),
                nn.Linear(in_planes, 1))

        else:
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, out_planes, bias=False),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True))
        
    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)

        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i-1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)

            x_knn, knn_idx = pointops.queryandgroup(
                self.nsample, p, n_p, x, None, o, n_o, use_xyz=True, return_idx=True)  # (m, nsample, 3+c)
            # knn_idx.shape = (m, nsample)
            # p_r = x_knn[:, :, :3] # (m, nsample, 3)
            # x_knn = x_knn[:, :, 3:] # (m, nsample, c)

            m, k, c = x_knn.shape
            x_knn_flatten = rearrange(x_knn, 'm k c -> (m k) c')
            x_knn_flatten_shrink = self.channel_shrinker(x_knn_flatten) # (m*nsample, 1)
            x_knn_shrink = rearrange(x_knn_flatten_shrink, '(m k) c -> m k c', m=m, k=k)
            x_knn_prob_shrink = F.softmax(x_knn_shrink, dim=1)

            y = self.linear2(x) # (n, c)
            with torch.no_grad():
                knn_idx_flatten = rearrange(knn_idx, 'm k -> (m k)')
            y_knn_flatten = y[knn_idx_flatten, :] # (m*nsample, c)
            y_knn = rearrange(y_knn_flatten, '(m k) c -> m k c', m=m, k=k)
            x_knn_weighted = y_knn * x_knn_prob_shrink # (m, nsample, c_out)
            y = torch.sum(x_knn_weighted, dim=1).contiguous() # (m, c_out)            
            p, o = n_p, n_o
        else:
            y = self.linear2(x)  # (n, c)

        return [p, y, o]
