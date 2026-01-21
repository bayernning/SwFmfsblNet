# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torch.nn as nn
import warnings
import time
import shutil
from math import ceil
import torch.optim as optim
from network.dispimage import convert_images_pt_to_png
from options import set_opts
import logging
import numpy as np
from network import Datasets, utils_logger, zidian, util
from network.cfmf_cdT import cfmf
from scipy.io import loadmat
import os
from scipy.io import savemat
from skimage.metrics import structural_similarity as compare_ssim
import gc
from network.losses import CompositeLoss, PearsonLoss
from network.train_utils import test_model, train_model, clean_memory, count_param, load_checkpoint
import matplotlib.pyplot as plt
# from network.classifier import test_classifier
import hdf5storage

if __name__ == "__main__":
    torch.cuda.empty_cache()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # 获取参数
    args = set_opts()

    # 模型参数设置
    ind_t = 2
    ps_list = [65, 0.5]
    # 修改前
    # clamp_list = [1e-3, 2]
    # 修改后 (匹配单精度训练的设置)
    clamp_list = [1e-6, 10.0]
    noise = 0

    # 目录设置
    args.model_dir = os.path.join('./model/model'+str(ind_t)+'/')
    args.log_dir = os.path.join('./log/')
    args.output_dir = "./output_data/"
    args.Test_dir=f"./test_data/mix_data1_76800_20dB.mat" 
    args.labelTest_dir='./test_data/test_label1_76800' 

    # 模型初始化参数
    cinit = np.random.rand(args.layer_num - 1) * 1e-5 + 1e-3
    dinit = np.ones(args.layer_num - 1) * 1e-4
    Tinit = np.ones(args.layer_num) * 0.34726566502035863

    Para = {
        'layer_num': args.layer_num,
        'Batch_size': args.batch_size,
        'P': args.Nr,
        'Q': args.Na,
        'N': args.N,
        'a0': 1e-5,
        'b0': 1e-5,
        'c0': cinit.data,
        'd0': dinit.data,
        'layernum': args.layer_num,
        'T': Tinit.data
    }

    # GPU设置
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    if isinstance(args.gpu_id, int):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in list(args.gpu_id))


    # ================ 数据加载 ================
    zidian, zidiancuda = zidian.create_zidian(args.G_dir, args.GG_dir)
    data_test = hdf5storage.loadmat(args.Test_dir)
    label_test = hdf5storage.loadmat(args.labelTest_dir)

    tic = time.time()
    datasets = {
        'test': Datasets.SimulateTest(data_test, label_test, zidian=zidian)
        # 'test': DenoisingDatasets.SimulateTest(data_test, label_test, zidian=zidian)
        
    }
    toc = time.time()
    print('{:.2f}'.format(toc - tic))

    # ================ 训练设置 ================
    print('\nBegin test with GPU: ' + str(args.gpu_id))

    batch_size = {
        'test': args.test_batchsize
    }

    data_loader = {
        phase: torch.utils.data.DataLoader(
            datasets[phase],
            batch_size=batch_size[phase],
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        ) for phase in datasets.keys()
    }

    num_data = {phase: len(datasets[phase]) for phase in datasets.keys()}
    num_iter_epoch = {phase: ceil(num_data[phase] / batch_size[phase]) for phase in datasets.keys()}

    # ================ 加载检查点 ================
    net = cfmf(Para).cuda()
    checkpoint = torch.load('./model/model8/model_epoch_50.pth','cuda')  
    criterion = CompositeLoss(weight_pearson=ps_list[0], weight_mse=ps_list[1], weight_l1=0.1)

    net.load_state_dict(checkpoint['model_state_dict'])  # 加载预训练的权重
    Loss, final_outputs, final_labels = test_model(net, data_loader, criterion, args, num_iter_epoch, zidiancuda, clamp_list, phase='test')

    save_path = os.path.join(args.output_dir, f'test_patch_{noise}dB.pt')
    torch.save({
        'image': final_outputs,
        'label': final_labels
        }, save_path)






