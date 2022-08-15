import os
import sys
from einops import rearrange

sys.path.append(os.path.abspath(".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from pointops2.functions import pointops
from classification_ModelNet40.models.pointmixer import (
    PointMixerBlockPaperInterSetLayerGroupMLPv3,
    PointMixerIntraSetLayerPaper,
)


class IdentityInterSetLayer(nn.Module):
    def __init__(self, in_planes, share_planes, nsample=16, use_xyz=False):
        super().__init__()
        self.in_planes = in_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.use_xyz = use_xyz

    def forward(self, input):
        x = input[0]
        return x


class PointNet2Layer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.mlp = nn.Sequential(
            nn.Linear(in_planes + 3, out_planes // 2, bias=False),
            Rearrange('n k c -> n c k'),
            nn.BatchNorm1d(out_planes // 2),
            Rearrange('n c k -> n k c'),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // 2, out_planes)
        )
        
    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_knn = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
        x_knn = self.mlp(x_knn)
        x = x_knn.max(dim=1)[0] # only values
        return x


class PointMixerIntraSetLayerPaperLinear(PointMixerIntraSetLayerPaper):
    expansion = 1

    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super(PointMixerIntraSetLayerPaperLinear, self).__init__(in_planes, out_planes, share_planes, nsample)

        self.channelMixMLPs01 = nn.Sequential( # input.shape = [N, K, C]
            nn.Linear(3+in_planes, nsample),
            nn.ReLU(inplace=True),
            nn.Linear(nsample, nsample)
        ) # overriding

#############################################################################################################

class PointMixerBlockPaperIdentityInterSetLayerGroupMLPv3(PointMixerBlockPaperInterSetLayerGroupMLPv3):
    interLayer = IdentityInterSetLayer


class PointNet2Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(PointNet2Block, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.layer2 = PointNet2Layer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.layer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]

#############################################################################################################

class Transformer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes, mid_planes // share_planes),
                                    nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                    nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
        x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
        x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # (n, nsample, c)
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, c)
        for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_v.shape; s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        #x = pointops.aggregation(x_v, w)
        return x


class TransitionDown(nn.Module):
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
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)  # (m, 3+c, nsample)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2*in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
        
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


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(Bottleneck, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = Transformer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]

##############################################################################################

class PointTransformerClsNet(nn.Module):
    def __init__(
        self, blocks, 
        c=3, k=40, nsample=[16,16,16,16,16], stride=[1,2,2,2,2],
        planes=[32, 64, 128, 256, 512],
        share_planes=8,
        transformerblock='Bottleneck',
        transup='TransitionUp',
        transdown='TransitionDown',
        use_avgmax=False,
    ):
        super().__init__()
        
        self.c = c
        self.transformerblock = transformerblock
        self.transup = transup
        self.transdown = transdown
        self.in_planes = c
        self.use_avgmax = use_avgmax
        
        assert stride[0] == 1, 'or you will meet errors.'

        self.enc1 = self._make_enc(planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/2
        self.enc3 = self._make_enc(planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/4
        self.enc4 = self._make_enc(planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/8
        self.enc5 = self._make_enc(planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])  # N/16
        self.emb_dim = planes[4]
        cls_in_planes = int(2 * planes[4]) if use_avgmax else planes[4] 
        self.cls = nn.Sequential(
            nn.Linear(cls_in_planes, planes[4]),
            nn.BatchNorm1d(planes[4]), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(planes[4], planes[4] // 2),
            nn.BatchNorm1d(planes[4] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(planes[4] // 2, k)
        )

    def _make_enc(self, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(globals()[self.transdown](
            self.in_planes, planes, stride, nsample))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(globals()[self.transformerblock](
                self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def po_from_batched_pcd(self, pcd):
        # x.shape: (B, 3, N)
        B, C, N = pcd.shape
        assert C == 3
        p = pcd.transpose(1, 2).contiguous().view(-1, 3) # (B*N, 3)
        o = torch.IntTensor([N * i for i in range(1, B + 1)]).to(p.device) # (N, 2N, ..)
        return (p, o)

    def forward(self, pcd):
        B = pcd.shape[0]
        p0, o0 = self.po_from_batched_pcd(pcd)
        x0 = p0

        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x5 = x5.view(B, -1, self.emb_dim).transpose(1, 2).contiguous()
        if self.use_avgmax:
            x5_avg = F.adaptive_avg_pool1d(x5, 1).squeeze(dim=-1)
            x5_max = F.adaptive_max_pool1d(x5, 1).squeeze(dim=-1)
            x5 = torch.cat([x5_avg, x5_max], dim=-1)
        else:
            x5 = F.adaptive_max_pool1d(x5, 1).squeeze(dim=-1)
        x5 = self.cls(x5).view(B, -1)
        return x5