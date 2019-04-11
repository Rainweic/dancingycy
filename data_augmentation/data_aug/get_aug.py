# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import argparse
import time

import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops import dataset_config
from torch.nn import functional as F
from ops.pre_stab_dataset import PreStabDataset
from configs.test_options import TestOptions
from configs.train_options import TrainOptions
from ops.stabnet import StabNetPre2, StabNet
import tensorboardX
from ops.utils import *
import os
import copy
import pickle
from ops.stn import STN
import cv2

path = './data/temp/'
opt = TrainOptions().parse()
data_loader = torch.utils.data.DataLoader(
        PreStabDataset(opt),
        batch_size=opt.batchSize, shuffle=True,
        num_workers=opt.nThreads)

for root, _, fnames in os.walk(path):
    for fname in fnames:
        if (not fname.endswith('.jpg')):
            continue
        name = fname[:-4]
        for i in range (10):
            t = data_loader.dataset.get_item2(os.path.join(root, fname))
            cv2.imwrite(os.path.join(path, name+'_'+str(i)+'.jpg'), tensor2im(t['unstab']))
