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

        im_gt = np.expand_dims(im_gt, 0).astype(np.complex128)
        y = np.expand_dims(y, 0).astype(np.complex128)
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

        im_gt = np.expand_dims(im_gt, 0).astype(np.complex128)
        y = np.expand_dims(y, 0).astype(np.complex128)

        return y, im_gt,index,idx


class SimulateTest_pd(Dataset):
    def __init__(self, data_test, SNR, A0, B0, Na0):
        super(SimulateTest_pd, self).__init__()
        self.data_test = data_test
        self.SNR = SNR
        self.A0 = A0
        self.B0 = B0
        self.Na0 = Na0

    def __len__(self):
        return (self.data_test['x'].shape[0])

    def __getitem__(self, index):
        im_gt1 = self.data_test['x'][index, :, :]
        echo = np.dot(np.dot(self.A0, im_gt1), self.B0)
        H, W = echo.shape

        # generate noise 
        np.random.seed(index)
        S = np.sum(np.power(np.abs(echo.squeeze()), 2)) / W / H
        SNRre = 10 ** (self.SNR / 10)
        noisePower = np.array([S / SNRre])
        #        noisePower = S/SNRre
        noise = np.sqrt(noisePower / 2) * (np.random.randn(H, W) + 1j * np.random.randn(H, W))
        echo_noisy = echo + noise
        eps = 1 / noisePower

        echo_noisy = np.delete(echo_noisy, self.Na0, axis=1)

        y = np.expand_dims(echo_noisy, 2).astype(np.complex128)
        x = np.expand_dims(im_gt1, 2).astype(np.complex128)
        y_test = torch.from_numpy(y.transpose((2, 0, 1)))
        x_test = torch.from_numpy(x.transpose((2, 0, 1)))

        eps = np.expand_dims(np.expand_dims(eps, 1), 2).astype(np.float64)
        eps = torch.from_numpy(eps.transpose((2, 0, 1)))

        # 加平动量
        B0 = 0.42e9
        fc = 10e9
        C = 3e8
        _, H1, W1 = y_test.shape

        torch.manual_seed(1)
        rm_true = (torch.rand(1, W1) * 2 - 1) * 5  # [1, 90]
        fr = (torch.linspace(-H1 / 2, H1 / 2 - 1, H1) * B0 / H1).unsqueeze(1)  # [128,1]
        #        phi = 2*math.pi*torch.rand(1,Na)-math.pi
        #        T = torch.exp(-1j*4*math.pi/self.C*torch.matmul(fr+self.fc, rm)+1j*phi)
        T = torch.exp(-1j * 4 * math.pi / C * torch.matmul(fr + fc, rm_true))
        S = y_test * T  # 带平动的回波  S: batch*1*Nr*Na

        return S, rm_true, y_test, x_test, eps
