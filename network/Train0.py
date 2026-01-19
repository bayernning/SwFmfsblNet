import torch
import torch.nn.functional as F
import torch.nn as nn
import warnings
import time
import shutil
from math import ceil
import torch.optim as optim
from options import set_opts
import logging
import numpy as np
from network import DenoisingDatasets, utils_logger, zidian, util
from network.cfmf_cdT import cfmf
from scipy.io import loadmat
import os
from scipy.io import savemat
from skimage.metrics import structural_similarity as compare_ssim
import gc
from network.losses import PearsonLoss
from network.train_utils import test_model, train_model, train_model2,clean_memory, count_param, load_checkpoint
# from network.classifier import test_classifier
import hdf5storage

# ================ 初始化设置 ================
torch.cuda.empty_cache()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 获取参数
args = set_opts()

# seed = 89228
# torch.manual_seed(seed)
# np.random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False



# 模型参数设置
ind_t = 12
ps_list = [80, 0.2]
clamp_list = [5e-3, 2]

# 目录设置
args.model_dir = os.path.join('./model/model'+str(ind_t)+'/')
args.log_dir = os.path.join('./log/')
args.output_dir = "./output_data/"
args.testoutput_dir = "./test_output" 

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

# ================ 模型初始化 ================
net = cfmf(Para).cuda()

# ================ 日志设置 ================
logger_name = 'log' + str(ind_t)

if os.path.isdir(args.log_dir):
    shutil.rmtree(args.log_dir)
os.makedirs(args.log_dir)
utils_logger.logger_info(logger_name, log_path=os.path.join(args.log_dir, logger_name + '.log'))
logger = logging.getLogger(logger_name)

if os.path.isdir(args.model_dir):
    shutil.rmtree(args.model_dir)
os.makedirs(args.model_dir)

# 记录参数信息
param = count_param(net)
logger.info('------> para = {:.2f}M: {:.2f}'.format(param / 1e6, param))
logger.info(f'ps_list: {ps_list}')
logger.info(f'clamp_list: {clamp_list}')

# 打印参数
for arg in vars(args):
    logger.info('------> {:<15s}: {:s}'.format(arg, str(getattr(args, arg))))


# 过滤警告
warnings.simplefilter('ignore', Warning, lineno=0)

# GPU设置
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
if isinstance(args.gpu_id, int):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in list(args.gpu_id))



# ================ 优化器设置 ================
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

# ================ 数据加载 ================
zidian, zidiancuda = zidian.create_zidian(args.G_dir, args.GG_dir)

data_train = hdf5storage.loadmat(args.Train_dir)
label_train = hdf5storage.loadmat(args.labelTrain_dir)
data_test = hdf5storage.loadmat(args.Test_dir)
label_test = hdf5storage.loadmat(args.labelTest_dir)

tic = time.time()
datasets = {
    'train': DenoisingDatasets.SimulateTrain(data_train, label_train, zidian=zidian),
    'test': DenoisingDatasets.SimulateTest(data_test, label_test, zidian=zidian)
}
toc = time.time()
print('{:.2f}'.format(toc - tic))

# ================ 训练设置 ================
print('\nBegin training with GPU: ' + str(args.gpu_id))

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
        pin_memory=True
    ) for phase in datasets.keys()
}

num_data = {phase: len(datasets[phase]) for phase in datasets.keys()}
num_iter_epoch = {phase: ceil(num_data[phase] / batch_size[phase]) for phase in datasets.keys()}

phase = "train"
criterion = PearsonLoss(weight_pearson=ps_list[0], weight_mse=ps_list[1])
Loss = torch.zeros(args.epochs)
total_time = 0

# 用于收集最后一个epoch的输出
final_outputs = []
final_labels = []

# 训练阶段
for epoch in range(args.epochs):
    # 训练阶段
    net.train()
    loss_per_epoch = 0
    mse_per_epoch = 0
    pearson_per_epoch = 0
    ssim_per_epoch = 0
    num_data = len(data_loader[phase].dataset)
    # 记录epoch开始时间
    epoch_start_time = time.time()
    
    output_numpy = []
    idx_all = []
    for ii, data in enumerate(data_loader[phase]):
        y_te, x_te, indices,idx = data
        y_te = y_te.cuda()
        x_te = x_te.cuda()
        
        optimizer.zero_grad()
        x_out, c0, d0, T = net(y_te, zidiancuda, clamp_list[0], clamp_list[1])
        
        # 保存最后一个epoch的输出
        
        if (epoch + 1) % 10 ==0 or epoch == 0:
            for i in range(len(indices)):
                output = x_out[i].detach().cpu().numpy()
                output_numpy.append(output)
                idx_all.append(idx[i].item())
            image = []
            label = []
            if len(output_numpy) == num_data:
                for i in range(int(num_data/args.N)):
                    start_idx = i * 256
                    end_idx = (i + 1) * 256
                    segment = output_numpy[start_idx:end_idx]
                    segment_tensors = [torch.tensor(s) if not isinstance(s, torch.Tensor) else s for s in segment]
                    concatenated_image = torch.cat(segment_tensors, dim=0)
                    image.append(concatenated_image)
                    label.append(idx_all[start_idx])

                    # 将收集的输出转换为张量
                final_labels = torch.tensor(label)
                final_outputs = torch.stack(image)
                final_outputs = final_outputs.squeeze(-1)

                save_path = os.path.join(args.output_dir, f'train_patch_{epoch + 1}.pt')
                torch.save({
                    'image': final_outputs,
                    'label': final_labels
                }, save_path)
                                    
        
        # 计算损失
        loss, pearson_iter, mse_iter = criterion(x_out, x_te)
        ssim_iter = util.batch_SSIM(abs(x_out), abs(x_te))
        loss.backward()
        optimizer.step()
        
        # 更新统计
        loss_per_epoch += loss.item() / num_iter_epoch[phase]
        ssim_per_epoch += ssim_iter / num_iter_epoch[phase]
        pearson_per_epoch += pearson_iter / num_iter_epoch[phase]
        mse_per_epoch += mse_iter/ num_iter_epoch[phase] 
        
        # 打印进度
        if (ii + 1) % np.floor(num_iter_epoch[phase] / 2) == 0:
            logger.info(
                '------> [Epoch:{:>2d}/{:<2d}] {:s}:{:0>4d}/{:0>4d}, Loss={:+.2e},Pearson={:+.4e}, lr={:.1e}'.format(
                    epoch + 1,
                    args.epochs,
                    phase,
                    ii + 1,
                    num_iter_epoch[phase],
                    loss_per_epoch,
                    pearson_per_epoch,
                    optimizer.param_groups[0]['lr']
                )
            )
    scheduler.step()
    # 计算epoch耗时
    epoch_time = time.time() - epoch_start_time
    total_time += epoch_time
    
    # 记录epoch结果
    logger.info(
        '------> [Epoch:{:>2d}/{:<2d}] {:s}: Loss={:+.4f} {:+.4f} {:+.4f} {:+.4f}, Time={:.2f}s'.format(
            epoch + 1,
            args.epochs,
            phase,
            loss_per_epoch,
            pearson_per_epoch,
            ssim_per_epoch,
            mse_per_epoch,
            epoch_time
        )
    )
    
    # 打印模型参数
    print(c0)
    print(d0)
    print(T)
    logger.info('-' * 50)
    
    # 保存损失     
    Loss[epoch] = loss_per_epoch
    
    # 保存检查点
    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(args.model_dir, f'model_epoch_{epoch + 1}.pth')
        torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
    
    
    # 清理内存
    gc.collect()
    torch.cuda.empty_cache()

# 打印总训练时间
logger.info(f'总训练时间: {total_time:.2f}秒')
logger.info(f'平均每个epoch时间: {total_time/args.epochs:.2f}秒')