#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-02 15:51:11

import torch
import numpy as np
# import scipy.io as io
# import random
import math
from torch.utils.data import Dataset


# Simulation Datasets:
class SimulateTrain(Dataset):
    def __init__(self, data_train, label_train, zidian=None):
        super(SimulateTrain, self).__init__()
        self.zidian = zidian
        self.data_train = data_train
        self.label_train = label_train

    def __len__(self):
        return len(self.data_train['data_train'][0])

    def __getitem__(self, index):
        # 读取标签图像
        im_gt = self.label_train['label_train'][0][index]['im_ori']  # 标签
        y = self.data_train['data_train'][0][index]['im_ori']

        # im_gt = np.expand_dims(im_gt, 0).astype(np.complex128)
        # y = np.expand_dims(y, 0).astype(np.complex128)
        im_gt = np.expand_dims(im_gt, 0).astype(np.complex64)
        y = np.expand_dims(y, 0).astype(np.complex64)
        idx = self.data_train['data_train'][0][index]['idx'][0][0]
        
        # 加入顺序标签 index
        return y, im_gt, index, idx


class SimulateTest(Dataset):
    def __init__(self, data_test, label_test, zidian=None):
        super(SimulateTest, self).__init__()
        self.zidian = zidian
        self.data_test = data_test
        self.label_test = label_test

    def __len__(self):
        return len(self.data_test['data_test'][0])

    def __getitem__(self, index):
        im_gt = self.label_test['label_test'][0][index]['im_ori']  # 标签
        y = self.data_test['data_test'][0][index]['im_ori']
        idx = self.data_test['data_test'][0][index]['idx'][0][0]

        # im_gt = np.expand_dims(im_gt, 0).astype(np.complex128)
        # y = np.expand_dims(y, 0).astype(np.complex128)
        im_gt = np.expand_dims(im_gt, 0).astype(np.complex64)
        y = np.expand_dims(y, 0).astype(np.complex64)


        return y, im_gt,index,idx

