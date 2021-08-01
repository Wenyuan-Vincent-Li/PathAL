## system package
import os, sys
sys.path.append('../')
import warnings
warnings.filterwarnings("ignore")
## general package
import torch
import torch.nn as nn
from collections import OrderedDict
from fastai.vision import *
## custom package
from utiles.mishactivation import Mish
from utiles.hubconf import *


class Model(nn.Module):
    def __init__(self, arch='resnext50_32x4d_ssl'):
        super().__init__()
        m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        self.head = nn.Sequential(AdaptiveConcatPool2d(), Flatten())

    def forward(self, x):
        """
        x: [bs, N, 3, h, w]
        x_out: [bs, N]
        """
        result = OrderedDict()
        x = self.enc(x)  # x: bs*N x C x 4 x 4
        y = self.head(x)  # x: bs x n
        result['out'] = y
        return result

if __name__ == "__main__":
    img = torch.rand([1, 3, 2 * 256, 2 * 256]).cuda()
    model = Model().cuda()
    output = model(img)
    print(output['out'].shape)