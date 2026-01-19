# -*- coding: utf-8 -*-
import os
import time
import shutil
import warnings
from math import ceil

import torch
import torch.optim as optim
import numpy as np
import logging
import hdf5storage
import matplotlib.pyplot as plt

from options import set_opts
from network import DenoisingDatasets, utils_logger, zidian, util
from network.cfmf_cdT import cfmf
from network.losses import PearsonLoss
from network.train_utils import test_model, train_model, train_model2, clean_memory, count_param, load_checkpoint


if __name__ == "__main__":
    # ================ 训练控制参数 ================
    TRAIN_PHASE = 'train'  # 'train' / 'resume' / 'visualization' / 'classification' / ...

    # ================ 初始化设置 ================
    warnings.simplefilter('ignore', Warning)
    args = set_opts()

    # reproducibility (optional)
    # seed = 89228
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # 模型参数设置
    ind_t = 12
    ps_list = [60, 0.45]
    clamp_list = [1e-5, 2]

    # 目录设置（可覆盖）
    args.Train_dir = "./mixdata/train/mix_data_76800_20dB.mat"
    args.labelTrain_dir = "./mixdata/train/mix_label_76800.mat"

    args.model_dir = os.path.join('./model/model' + str(ind_t) + '/')
    args.log_dir = os.path.join('./log/')
    args.output_dir = "./output_data/"
    args.testoutput_dir = "./test_output"

    zidian_obj, zidiancuda = zidian.create_zidian(args.G_dir, args.GG_dir)

    # ================ Para 初始化（不包含 c0/d0/T，全部由 HyperNet 生成） ================
    Para = {
        'layer_num': args.layer_num,
        'Batch_size': args.batch_size,
        'zidiancuda': zidiancuda,
        'P': args.Nr,
        'Q': args.Na,
        'N': args.N,
        'a0': 1e-4,
        'b0': 1e-4
    }

    # ================ 设备配置 ================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    if isinstance(args.gpu_id, int):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in list(args.gpu_id))

    # ================ 模型初始化 ================
    net = cfmf(Para).to(device)
    # net = torch.compile(net)

    # ================ 日志与保存目录 ================
    logger_name = 'log' + str(ind_t)
    if TRAIN_PHASE == 'train':
        utils_logger.logger_info(logger_name, log_path=os.path.join(args.log_dir, logger_name + '.log'))
        logger = logging.getLogger(logger_name)

        if os.path.isdir(args.model_dir):
            shutil.rmtree(args.model_dir)
        os.makedirs(args.model_dir, exist_ok=True)

        # 记录参数信息
        param = count_param(net)
        logger.info('------> para = {:.2f}M: {:.2f}'.format(param / 1e6, param))
        logger.info(f'ps_list: {ps_list}')
        logger.info(f'clamp_list: {clamp_list}')
        for arg in vars(args):
            logger.info('------> {:<15s}: {:s}'.format(arg, str(getattr(args, arg))))
    else:
        logger = None
        os.makedirs(args.model_dir, exist_ok=True)

    # ================ 优化器设置 ================
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    # ================ 数据加载（使用 hdf5storage 读取 .mat） ================
    

    data_train = hdf5storage.loadmat(args.Train_dir)
    label_train = hdf5storage.loadmat(args.labelTrain_dir)
    data_test = hdf5storage.loadmat(args.Test_dir)
    label_test = hdf5storage.loadmat(args.labelTest_dir)

    tic = time.time()
    datasets = {
        'train': DenoisingDatasets.SimulateTrain(data_train, label_train, zidian=zidian_obj),
        'test': DenoisingDatasets.SimulateTest(data_test, label_test, zidian=zidian_obj)
    }
    toc = time.time()
    print('Dataset build time: {:.2f}s'.format(toc - tic))

    # ================ 训练设置 ================
    print('\nBegin training on device: {}'.format(device))

    batch_size = {
        'train': args.batch_size,
        'test': args.test_batchsize
    }

    data_loader = {
        phase: torch.utils.data.DataLoader(
            datasets[phase],
            batch_size=batch_size[phase],
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0)
        ) for phase in datasets.keys()
    }

    num_data = {phase: len(datasets[phase]) for phase in datasets.keys()}
    num_iter_epoch = {phase: ceil(num_data[phase] / batch_size[phase]) for phase in datasets.keys()}

    # ================ 加载检查点辅助函数 ================
    def load_saved_patches(data_dir, epoch):
        checkfile = os.path.join(data_dir, f'train_patch_{epoch}.pt')
        data = torch.load(checkfile, map_location='cpu')
        final_outputs = data['image']
        final_labels = data['label']
        return final_outputs, final_labels

    # ================ 训练 / 分类 流程 ================
    if TRAIN_PHASE == 'train':
        criterion = PearsonLoss(weight_pearson=ps_list[0], weight_mse=ps_list[1])

        # train_model2: 保持你已有接口（net, data_loader, optimizer, scheduler, criterion, args, logger, ...)
        Loss, final_outputs, final_labels = train_model2(net, data_loader, optimizer, scheduler,criterion, args, 
                                                         logger, num_iter_epoch, zidiancuda, clamp_list, start_epoch=0, phase='train')

        
        final_outputs, final_labels = load_saved_patches(args.output_dir, 100)
           

    # 清理内存
    clean_memory()
