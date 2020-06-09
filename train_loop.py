from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from dataset import MultiResDataset
from DataIterator import DataIterator

from utils import set_lr

"""
args에서 받아야할 파라미터
root_dir
transforms
llvl: 시작 l
minibatch: dictionary 형식 { resolution: minibatch_size, ...}
num_workers: 데이터로딩할 때 사용할 worker 수
"""

# 여기서 G D 만들고...


def train(args):
    dataloader = DataIterator(
        MultiResDataset(
            args.root_dir, args.transforms,
            args.llvl), args.minibatch[2 ** args.lvll],
        shuffle=True,
        num_workers=args.num_workers)

    G_opt = optim.Adam(**args.G_opt_param, lr=args.G_opt_lr[2 ** args.llvl])
    D_opt = optim.Adam(**args.D_opt_param, lr=args.D_opt_lr[2 ** args.llvl])

    nimg = 0
    while nimg < args.total_imgs:
        for minibatch_i in range(args.mb_repeat):
            batch = next(dataloader)
            
        nimg += args.mb_repeat * args.minibatch

