# -*- coding: utf-8 -*-
import os
import time
import warnings
from math import ceil
from datetime import datetime

import torch
import torch.optim as optim
import numpy as np
import logging
import hdf5storage
import matplotlib.pyplot as plt

from options import set_opts
from network import Datasets, utils_logger, zidian, util
from network.cfmf_cdT import cfmf
from network.losses import CompositeLoss, PearsonLoss
from network.train_utils import test_model, train_model, train_model2, clean_memory, count_param, load_checkpoint


if __name__ == "__main__":
    # ================ 训练控制参数 ================
    TRAIN_PHASE = 'train'  # 'train' / 'resume' / 'visualization' / 'classification' / ...

    # ================ 初始化设置 ================
    warnings.simplefilter('ignore', Warning)
    args = set_opts()

    # 生成运行时间戳，用于日志和模型目录命名
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # reproducibility (optional)
    # seed = 89228
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # 模型参数设置
    ps_list = [60, 0.45]
    # 修改前
    # clamp_list = [1e-5, 2]

    # 修改后 (放宽上限，下限保持 1e-5 或 1e-6 均可)
    clamp_list = [1e-6, 10.0]

    # 目录设置（可覆盖）
    args.Train_dir = "./train_data/mix_data_76800_20dB.mat"
    args.labelTrain_dir = "./train_data/mix_label_76800.mat"

    # 使用运行时间戳创建日志和模型目录
    args.model_dir = os.path.join('./model', f'model_{run_timestamp}/')
    args.log_dir = os.path.join('./log', f'log_{run_timestamp}/')
    args.output_dir = "./output_data/"
    args.testoutput_dir = "./test_output"

    # 自动创建目录
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.testoutput_dir, exist_ok=True)

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
    logger_name = f'log_{run_timestamp}'
    if TRAIN_PHASE == 'train':
        utils_logger.logger_info(logger_name, log_path=os.path.join(args.log_dir, f'{logger_name}.log'))
        logger = logging.getLogger(logger_name)

        # 记录参数信息
        param = count_param(net)
        logger.info('------> para = {:.2f}M: {:.2f}'.format(param / 1e6, param))
        logger.info(f'ps_list: {ps_list}')
        logger.info(f'clamp_list: {clamp_list}')
        for arg in vars(args):
            logger.info('------> {:<15s}: {:s}'.format(arg, str(getattr(args, arg))))
    else:
        logger = None

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
        'train': Datasets.SimulateTrain(data_train, label_train, zidian=zidian_obj),
        'test': Datasets.SimulateTest(data_test, label_test, zidian=zidian_obj)
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
        # criterion = PearsonLoss(weight_pearson=ps_list[0], weight_mse=ps_list[1])
        # 2. 实例化这个新的 Loss
        criterion = CompositeLoss(
            weight_pearson=1000.0, 
            weight_mse=0.001, 
            weight_l1=0.001,      
            weight_smooth=0.005   
        )

        # train_model2: 保持你已有接口（net, data_loader, optimizer, scheduler, criterion, args, logger, ...)
        Loss, final_outputs, final_labels = train_model2(net, data_loader, optimizer, scheduler,criterion, args, 
                                                         logger, num_iter_epoch, zidiancuda, clamp_list, start_epoch=0, phase='train')

        
        # final_outputs, final_labels = load_saved_patches(args.output_dir, 100)
           

    # 清理内存
    clean_memory()
