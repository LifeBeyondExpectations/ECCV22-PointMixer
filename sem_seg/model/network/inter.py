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


class BilinearFeedForward(nn.Module):

    def __init__(self, in_planes1, in_planes2, out_planes):
        super().__init__()
        self.bilinear = nn.Bilinear(in_planes1, in_planes2, out_planes)

    def forward(self, x):
        x = x.contiguous()
        x = self.bilinear(x, x)
        return x

####################################################################################

class NoInterSetLayer(nn.Module):
    def __init__(self, in_planes, nsample=16, use_xyz=False):
        super().__init__()
    
    def forward(self, input):
        x, x_knn, knn_idx, p_r = input
        return x

# PointMixerInterSetLayerV3
class PointMixerInterSetLayer(nn.Module):

    def __init__(self, in_planes, share_planes, nsample):
        super().__init__()
        self.share_planes = share_planes
        self.nsample = nsample

        self.linear = nn.Sequential(
            nn.Linear(in_planes+in_planes, in_planes//share_planes), # [N*K, C] 
            nn.ReLU(inplace=True))
        self.linear_x = nn.Sequential(
            nn.Linear(in_planes, in_planes//share_planes), # [N*K, C]
            nn.ReLU(inplace=True))
        self.linear_p = nn.Sequential( # [N*K, C]
            nn.Linear(3, 3, bias=False),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True), 
            nn.Linear(3, in_planes))

    def forward(self, input):
        x, x_knn, knn_idx, p_r = input
        # p_r: (n, nsample, 3) # point relative (p_r) 
        # x_knn: (n, nsample, c) 
        # knn_idx: (n, nsample)
        N = x_knn.shape[0]

        with torch.no_grad():
            knn_idx_flatten = rearrange(knn_idx, 'n k -> (n k) 1')
        p_r_flatten = rearrange(p_r, 'n k c -> (n k) c')
        p_embed_flatten = self.linear_p(p_r_flatten)
        x_knn_flatten = rearrange(x_knn, 'n k c -> (n k) c')
        x_knn_flatten_shrink = self.linear(
            torch.cat([p_embed_flatten, x_knn_flatten], dim=1))

        x_knn_prob_flatten_shrink = \
            scatter_softmax(x_knn_flatten_shrink, knn_idx_flatten, dim=0) # (n*nsample, c')
        x_v_knn_flatten = self.linear_x(x_knn_flatten) # (n*nsample, c')
        x_knn_weighted_flatten = x_v_knn_flatten * x_knn_prob_flatten_shrink # (n*nsample, c')

        residual = scatter_sum(x_knn_weighted_flatten, knn_idx_flatten, dim=0, dim_size=N) # (n, c')
        residual = repeat(residual, 'n c -> n (repeat c)', repeat=self.share_planes)
        return x + residual
