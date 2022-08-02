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
from .hier import *
from .inter import *

seed=0
# pl.seed_everything(seed, workers=True)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if use multi-GPU
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

class PointTransformerIntraSetLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(            
            nn.Linear(3, 3),
            nn.Sequential(
                Rearrange('n k c -> n c k'),
                nn.BatchNorm1d(3),
                Rearrange('n c k -> n k c')),
            nn.ReLU(inplace=True), 
            nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(
            nn.Sequential(
                Rearrange('n k c -> n c k'),
                nn.BatchNorm1d(mid_planes), 
                Rearrange('n c k -> n k c')),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, mid_planes // share_planes),
            nn.Sequential(
                Rearrange('n k c -> n c k'),
                nn.BatchNorm1d(mid_planes // share_planes),
                Rearrange('n c k -> n k c')),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)

        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
        channel_length = x_k.shape[1]
        x_kv = torch.cat([x_k, x_v], dim=1)
        x_kv, knn_idx = pointops.queryandgroup(
            self.nsample, p, p, x_kv, None, o, o, use_xyz=True, return_idx=True)  # (n, nsample, 3+c)
        p_r = x_kv[:, :, 0:3]
        x_k = x_kv[:, :, 3:(3+channel_length)]
        x_v = x_kv[:, :, -channel_length:]

        p_embed = self.linear_p(p_r)
        w = x_k - x_q.unsqueeze(1) + p_embed.view(
            p_embed.shape[0], 
            p_embed.shape[1], 
            self.out_planes // self.mid_planes, 
            self.mid_planes).sum(2)  # (n, nsample, c)

        w = self.linear_w(w)

        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_v.shape; 
        s = self.share_planes
        
        x_knn = (x_v + p_embed).view(n, self.nsample, s, c // s)
        x_knn = (x_knn * w.unsqueeze(2))
        x_knn = x_knn.reshape(n, self.nsample, c)

        x = x_knn.sum(1)
        return (x, x_knn, knn_idx, p_r)

class PointTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, 
                 in_planes, planes, share_planes=8, 
                 nsample=16, 
                 use_xyz=False,
                 intraLayer='PointTransformerIntraSetLayer',
                 interLayer='NoInterSetLayer'):
        super().__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = nn.Sequential(
            globals()[intraLayer](planes, planes, share_planes, nsample),
            globals()[interLayer](in_planes, share_planes, nsample))
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes*self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x = x + identity
        x = self.relu(x)
        return [p, x, o]

class PointTransformerSegNet(nn.Module):
    mixerblock = PointTransformerBlock

    def __init__(
        self, block, blocks, 
        c=6, k=13, nsample=[8,16,16,16,16], stride=[1,4,4,4,4],
        intraLayer='PointTransformerIntraSetLayer',
        interLayer='NoInterSetLayer',
        transup='TransitionUp', 
        transdown='TransitionDownBlock'):
        super().__init__()
        
        self.c = c
        self.intraLayer = intraLayer
        self.interLayer = interLayer
        self.transup = transup
        self.transdown = transdown
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        
        assert stride[0] == 1, 'or you will meet errors.'

        self.enc1 = self._make_enc(planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])  # N/256
        self.dec5 = self._make_dec(planes[4], 2, share_planes, nsample=nsample[4], is_head=True)  # transform p5
        self.dec4 = self._make_dec(planes[3], 2, share_planes, nsample=nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1
        self.cls = nn.Sequential(
            nn.Linear(planes[0], planes[0]), 
            nn.BatchNorm1d(planes[0]), 
            nn.ReLU(inplace=True), 
            nn.Linear(planes[0], k))

    def _make_enc(self, planes, blocks, share_planes, stride, nsample):
        layers = []
        layers.append(globals()[self.transdown]( 
            in_planes=self.in_planes, 
            out_planes=planes, 
            stride=stride, 
            nsample=nsample))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(self.mixerblock(
                in_planes=self.in_planes, 
                planes=self.in_planes, 
                share_planes=share_planes,
                nsample=nsample,
                intraLayer=self.intraLayer,
                interLayer=self.interLayer))
        return nn.Sequential(*layers)

    def _make_dec(self, planes, blocks, share_planes, nsample, is_head=False):
        layers = []
        layers.append(globals()[self.transup](
            in_planes=self.in_planes, 
            out_planes=None if is_head else planes, 
            nsample=nsample))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(self.mixerblock(
                in_planes=self.in_planes, 
                planes=self.in_planes, 
                share_planes=share_planes,
                nsample=nsample,
                intraLayer=self.intraLayer,
                interLayer=self.interLayer))
        return nn.Sequential(*layers)

    def forward(self, pxo):
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        x = self.cls(x1)
        return x

def getPointTransformerSegNet(**kwargs):
    '''
    kwargs['transup'] = 'TransitionUpBlock'
    '''
    model = PointTransformerSegNet(PointTransformerBlock, [2, 3, 4, 6, 3], **kwargs)
    return model





