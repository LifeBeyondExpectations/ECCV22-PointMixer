import os
import sys

sys.path.append(os.path.abspath(".."))

from classification_ModelNet40.models.pointtransformer import PointTransformerClsNet


def pointTransformerBase(num_classes=40, **kwargs):
    return PointTransformerClsNet(
        [2, 3, 4, 6, 3], k=num_classes,
        nsample=[8,8,8,8,8],
        stride=[1,2,2,2,2],
        **kwargs
    )


def pointMixerBase(num_classes=40, **kwargs):
    return PointTransformerClsNet(
        [2, 3, 4, 6, 3], k=num_classes,
        nsample=[8,8,8,8,8],
        stride=[1,2,2,2,2],
        transformerblock='PointMixerBlockPaperIdentityInterSetLayerGroupMLPv3',
        **kwargs
    )


def pointNet2Base(num_classes=40, **kwargs):
    return PointTransformerClsNet(
        [2, 3, 4, 6, 3], k=num_classes,
        nsample=[8,8,8,8,8],
        stride=[1,2,2,2,2],
        transformerblock='PointNet2Block',
        **kwargs
    )


def pointMixerBaseLinear(num_classes=40, **kwargs):
    return PointTransformerClsNet(
        [2, 3, 4, 6, 3], k=num_classes,
        nsample=[8,8,8,8,8],
        stride=[1,2,2,2,2],
        transformerblock='PointMixerBlockPaperIdentityInterSetLayerGroupMLPv3Linear',
        **kwargs
    )