import os
import sys

sys.path.append(os.path.abspath(".."))

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from torch_scatter import scatter_softmax, scatter_sum

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointops2.functions import pointops


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


class BilinearFeedForward(nn.Module):

    def __init__(self, in_planes1, in_planes2, out_planes):
        super().__init__()
        self.bilinear = nn.Bilinear(in_planes1, in_planes2, out_planes)

    def forward(self, x):
        x = x.contiguous()
        x = self.bilinear(x, x)
        return x

##############################################################################################

class PointMixerIntraSetLayerPaper(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample

        self.channelMixMLPs01 = nn.Sequential( # input.shape = [N, K, C]
            nn.Linear(3+in_planes, nsample),
            nn.ReLU(inplace=True),
            BilinearFeedForward(nsample, nsample, nsample))
        
        self.linear_p = nn.Sequential( # input.shape = [N, K, C]
            nn.Linear(3, 3),
            nn.Sequential(
                Rearrange('n k c -> n c k'),
                nn.BatchNorm1d(3),
                Rearrange('n c k -> n k c')),
            nn.ReLU(inplace=True), 
            nn.Linear(3, out_planes))
        self.shrink_p = nn.Sequential(
            Rearrange('n k (a b) -> n k a b', b=nsample),
            Reduce('n k a b -> n k b', 'sum', b=nsample))

        self.channelMixMLPs02 = nn.Sequential( # input.shape = [N, K, C]
            Rearrange('n k c -> n c k'),
            nn.Conv1d(nsample+nsample, mid_planes, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_planes), 
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_planes, mid_planes//share_planes, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_planes//share_planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_planes//share_planes, out_planes//share_planes, kernel_size=1),
            Rearrange('n c k -> n k c'))

        self.channelMixMLPs03 = nn.Linear(in_planes, out_planes)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)

        x_knn, knn_idx = pointops.queryandgroup(
            self.nsample, p, p, x, None, o, o, use_xyz=True, return_idx=True)  # (n, k, 3+c)
        p_r = x_knn[:, :, 0:3]

        energy = self.channelMixMLPs01(x_knn) # (n, k, k)
        
        p_embed = self.linear_p(p_r) # (n, k, out_planes)
        p_embed_shrink = self.shrink_p(p_embed) # (n, k, k)

        energy = torch.cat([energy, p_embed_shrink], dim=-1)
        energy = self.channelMixMLPs02(energy) # (n, k, out_planes/share_planes)
        w = self.softmax(energy)

        x_v = self.channelMixMLPs03(x)  # (n, in_planes) -> (n, k)
        n = knn_idx.shape[0]; knn_idx_flatten = knn_idx.flatten()
        x_v  = x_v[knn_idx_flatten, :].view(n, self.nsample, -1)

        n, nsample, out_planes = x_v.shape
        x_knn = (x_v + p_embed).view(n, nsample, self.share_planes, out_planes//self.share_planes)
        x_knn = (x_knn * w.unsqueeze(2))
        x_knn = x_knn.reshape(n, nsample, out_planes)

        x = x_knn.sum(1)
        return (x, x_knn, knn_idx, p_r)


class PointMixerIntraSetLayerPaperv3(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample

        self.channelMixMLPs01 = nn.Sequential( # input.shape = [N*K, C]
            nn.Linear(3+in_planes, nsample),
            nn.ReLU(inplace=True),
            BilinearFeedForward(nsample, nsample, nsample))
        
        self.linear_p = nn.Sequential( # input.shape = [N*K, C]
            nn.Linear(3, 3),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True), 
            nn.Linear(3, out_planes))
        self.shrink_p = nn.Sequential(
            Rearrange('n k (a b) -> n k a b', b=nsample),
            Reduce('n k a b -> (n k) b', 'sum', b=nsample))

        self.channelMixMLPs02 = nn.Sequential( # input.shape = [N*K, C]
            nn.Linear(nsample+nsample, mid_planes, bias=False),
            nn.BatchNorm1d(mid_planes), 
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, mid_planes//share_planes, bias=False),
            nn.BatchNorm1d(mid_planes//share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes//share_planes, out_planes//share_planes, bias=True),
            Rearrange('(n k) c -> n k c', k=nsample))

        self.channelMixMLPs03 = nn.Linear(in_planes, out_planes)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)

        x_knn, knn_idx = pointops.queryandgroup(
            self.nsample, p, p, x, None, o, o, use_xyz=True, return_idx=True)  # (n, k, 3+c)
        p_r = x_knn[:, :, 0:3]
        
        x_knn_flatten = rearrange(x_knn, 'n k c -> (n k) c')
        energy_flatten = self.channelMixMLPs01(x_knn_flatten) # (n*k, k)
        
        n = p_r.shape[0]; 
        p_embed = self.linear_p(p_r.view(-1, 3)) # (n*k, out_planes)
        p_embed = p_embed.view(n, self.nsample, -1)
        p_embed_shrink_flatten = self.shrink_p(p_embed) # (n*k, k)

        energy_flatten = torch.cat([energy_flatten, p_embed_shrink_flatten], dim=-1) # (n*k, 2k)
        energy = self.channelMixMLPs02(energy_flatten) # (n, k, out_planes/share_planes)
        w = self.softmax(energy)

        x_v = self.channelMixMLPs03(x)  # (n, in_planes) -> (n, k)
        n = knn_idx.shape[0]; knn_idx_flatten = knn_idx.flatten()
        x_v  = x_v[knn_idx_flatten, :].view(n, self.nsample, -1)

        n, nsample, out_planes = x_v.shape
        x_knn = (x_v + p_embed).view(n, nsample, self.share_planes, out_planes//self.share_planes)
        x_knn = (x_knn * w.unsqueeze(2))
        x_knn = x_knn.reshape(n, nsample, out_planes)

        x = x_knn.sum(1)
        return (x, x_knn, knn_idx, p_r)


######################################################################################################

class PointMixerInterSetLayerGroupMLPv3(nn.Module):

    def __init__(self, in_planes, share_planes, nsample=16, use_xyz=False):
        super().__init__()
        self.share_planes = share_planes
        self.linear = nn.Linear(in_planes, in_planes//share_planes) # input.shape = [N*K, C] 
        self.linear_x = nn.Linear(in_planes, in_planes//share_planes) # input.shape = [N*K, C]
        self.linear_p = nn.Sequential( # input.shape = [N*K, C]
            nn.Linear(3, 3, bias=False),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True), 
            nn.Linear(3, in_planes))

    def forward(self, input):
        x, x_knn, knn_idx, p_r = input
        N = x_knn.shape[0]

        with torch.no_grad():
            knn_idx_flatten = rearrange(knn_idx, 'n k -> (n k) 1')
        p_r_flatten = rearrange(p_r, 'n k c -> (n k) c')
        p_embed_flatten = self.linear_p(p_r_flatten)
        x_knn_flatten = rearrange(x_knn, 'n k c -> (n k) c')
        x_knn_flatten_shrink = self.linear(x_knn_flatten + p_embed_flatten) # nk c'

        x_knn_prob_flatten_shrink = \
            scatter_softmax(x_knn_flatten_shrink, knn_idx_flatten, dim=0) # (n*nsample, c')
        x_v_knn_flatten = self.linear_x(x_knn_flatten) # (n*nsample, c')
        x_knn_weighted_flatten = x_v_knn_flatten * x_knn_prob_flatten_shrink # (n*nsample, c')

        residual = scatter_sum(x_knn_weighted_flatten, knn_idx_flatten, dim=0, dim_size=N) # (n, c')
        residual = repeat(residual, 'n c -> n (repeat c)', repeat=self.share_planes)
        return x + residual

###########################################################################

class PointMixerBlock(nn.Module):
    expansion = 1
    intraLayer = None
    interLayer = None

    def __init__(self, in_planes, planes, share_planes=8, nsample=16, use_xyz=False):
        assert self.intraLayer is not None
        assert self.interLayer is not None
        super().__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = nn.Sequential(
            self.intraLayer(planes, planes, share_planes, nsample),
            self.interLayer(in_planes, nsample, share_planes)
        )
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


class PointMixerBlockPaperInterSetLayerGroupMLPv3(PointMixerBlock):
    expansion = 1
    intraLayer = PointMixerIntraSetLayerPaper
    interLayer = PointMixerInterSetLayerGroupMLPv3

##############################################################################################

class SymmetricTransitionUpBlock(nn.Module):
    def __init__(self, in_planes, out_planes=None, nsample=16):
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
            self.linear1 = nn.Sequential( # input.shape = [N, L]
                nn.Linear(out_planes, out_planes), 
                nn.BatchNorm1d(out_planes),  
                nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential( # input.shape = [N, L]
                nn.Linear(in_planes, out_planes), 
                nn.BatchNorm1d(out_planes), 
                nn.ReLU(inplace=True))
            self.channel_shrinker = nn.Sequential( # input.shape = [N*K, L]
                nn.Linear(in_planes+3, in_planes),
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
            p1, x1, o1 = pxo1; p2, x2, o2 = pxo2 
            knn_idx = pointops.knnquery(self.nsample, p1, p2, o1, o2)[0].long()

            with torch.no_grad():
                knn_idx_flatten = rearrange(knn_idx, 'm k -> (m k)')
            p_r = p1[knn_idx_flatten, :].view(len(p2), self.nsample, 3) - p2.unsqueeze(1)
            x2_knn = x2.view(len(p2), 1, -1).repeat(1, self.nsample, 1)
            x2_knn = torch.cat([p_r, x2_knn], dim=-1) # (109, 16, 259) # (m, nsample, 3+c)

            with torch.no_grad():
                knn_idx_flatten = knn_idx_flatten.unsqueeze(-1) # (m*nsample, 1)
            x2_knn_flatten = rearrange(x2_knn, 'm k c -> (m k) c') # c = 3+out_planes
            x2_knn_flatten_shrink = self.channel_shrinker(x2_knn_flatten) # (m, nsample, 1)
            x2_knn_prob_flatten_shrink = scatter_softmax(x2_knn_flatten_shrink, knn_idx_flatten, dim=0)

            x2_knn_prob_shrink = rearrange(x2_knn_prob_flatten_shrink, '(m k) 1 -> m k 1', k=self.nsample)
            up_x2_weighted = self.linear2(x2).unsqueeze(1) * x2_knn_prob_shrink # (m, nsample, c)
            up_x2_weighted_flatten = rearrange(up_x2_weighted, 'm k c -> (m k) c')
            up_x2 = scatter_sum(up_x2_weighted_flatten, knn_idx_flatten, dim=0, dim_size=len(p1))
            y = self.linear1(x1) + up_x2
        return y

##############################################################################################

class SymmetricTransitionDownBlockPaperv3(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
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

##############################################################################################

class PointMixerClsNet(nn.Module):
    def __init__(
        self, blocks, 
        c=3, k=40, nsample=[16,16,16,16,16], stride=[1,2,2,2,2],
        planes=[32, 64, 128, 256, 512],
        share_planes=8,
        mixerblock='PointMixerBlockPaperInterSetLayerGroupMLPv3',
        transup='SymmetricTransitionUpBlock',
        transdown='SymmetricTransitionDownBlockPaperv3',
        use_avgmax=False,
    ):
        super().__init__()
        
        self.c = c
        self.mixerblock = mixerblock
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
            layers.append(globals()[self.mixerblock](
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


def pointMixerFinal(num_classes=40, **kwargs):
    return PointMixerClsNet(
        [2, 3, 4, 6, 3], k=num_classes,
        nsample=[8,8,8,8,8],
        stride=[1,2,2,2,2],
        use_avgmax=True,
        **kwargs
    )


if __name__ == '__main__':
    data = torch.rand(2, 3, 1024).cuda()
    print("===> testing pointMixer (cuda)...")
    model = pointMixerFinal().cuda()
    out = model(data)
    print(out.shape)
