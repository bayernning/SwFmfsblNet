# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:33:39 2024
HyperNet--FMF1D
@author: Dell
"""
import torch
import torch as tc
import torch.nn as nn
from scipy.io import savemat
import numpy as np
import torch.nn.functional as F


class HyperNet_1(torch.nn.Module):
    def __init__(self, kernel, pad, layer):
        super(HyperNet_1, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)  # 8 32 256
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)  # 8 32 64
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)  # 8 32 64
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)  # 8 32 16
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, padding=1)  # 8 32 16
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)  # 8 32 4
        # self.POOL = nn.AdaptiveMaxPool1d(3*20-2) # 8 32 1
        #        self.fc1 = nn.Linear(32*8*4,64)
        # self.fc2 = nn.Linear(32 * 4, 3 * layer - 2)
        self.fc_c = nn.Linear(32 * 4 * 1, layer - 1)
        self.fc_d = nn.Linear(32 * 4 * 1, layer - 1)
        self.fc_T = nn.Linear(32 * 4 * 1, layer)

    def forward(self, x, clamp_min, clamp_max):
        x = self.conv1(x)
        x = self.pool1(nn.functional.relu(x))
        x = self.conv2(x)
        x = self.pool2(nn.functional.relu(x))
        x = self.pool3(nn.functional.relu(self.conv3(x)))
        # x = self.pool4(nn.functional.relu(self.conv4(x)))
        # x = self.POOL(x)
        x = x.view(-1, 32 * 4)
        #        x = nn.functional.relu(self.fc1(x))
        # x = 2*nn.functional.sigmoid(self.fc2(x))

        # x = torch.abs(self.fc2(x))
        # x = x.view(-1, 1, 1, 3)
        c0 = tc.abs(self.fc_c(x))
        d0 = tc.abs(self.fc_d(x))
        T0 = tc.abs(self.fc_T(x))
        #        x = tc.abs(self.fc1(x)).clamp_(1e-6, 1)  # [batch 58]
        return c0, d0, T0


class HyperNet_7convL_CU_5_1(nn.Module):  # 通道数64  类似膨胀卷积，膨胀系数为3，卷积核4x4为只有最左侧列有值，丢失一部分细节信息，但比maxpool轻微
    def __init__(self, kernel, pad, layer_num):
        super(HyperNet_7convL_CU_5_1, self).__init__()
        # 升维
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)

        # CPMB1
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, padding=1, stride=2)
        self.conv32 = nn.Conv1d(32, 32, kernel_size=5, padding=1, stride=4)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 8 32 64

        # CPMB2
        self.conv4 = nn.Conv1d(32, 32, kernel_size=3, padding=1, stride=2)
        self.conv5 = nn.Conv1d(32, 32, kernel_size=3, padding=1, stride=2)
        self.conv52 = nn.Conv1d(32, 32, kernel_size=5, padding=1, stride=4)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # CPMB3
        # self.conv6 = nn.Conv1d(32, 32, kernel_size=kernel, padding=pad)
        # self.conv7 = nn.Conv1d(32, 32, kernel_size=1, stride=4)
        # self.conv72 = nn.Conv1d(32, 32, kernel_size=1, stride=4)

        # 升通道降维
        # self.conv8 = nn.Conv1d(32, 64, kernel_size=1, stride=1)

        #        self.fc1 = nn.Linear(32 * 4 * 1, 3 * layer_num - 2)
        self.fc_c = nn.Linear(32 * 4 * 1, layer_num - 1)
        self.fc_d = nn.Linear(32 * 4 * 1, layer_num - 1)
        self.fc_T = nn.Linear(32 * 4 * 1, layer_num)

    def forward(self, x, clamp_min, clamp_max):  # CRC-CRC-CRC-CR
        x = self.conv1(x)  # [8 32 256]

        x_1 = self.conv3(nn.functional.relu(self.conv2(x)))  # [8,32,64]
        # x_temp = x_1 * r.reshape(x_1.shape[0], -1, 1, 1)
        x_add = x_1 + self.conv32(x)  # [8,32,64]
        x = self.pool1(nn.functional.relu(x_add))  # [8 32 32]

        x_1 = self.conv5(nn.functional.relu(self.conv4(x)))  # [8,32,8]
        # x_temp = x_1 * r.reshape(x_1.shape[0], -1, 1, 1)
        x_add = x_1 + self.conv52(x)  # [8,32,8]
        x = self.pool2(nn.functional.relu(x_add))  # [8 32 4]

        # x_1 = self.conv7(nn.functional.relu(self.conv6(x)))  # [8,32,2,2]
        # # x_temp = x_1 * r.reshape(x_1.shape[0], -1, 1, 1)
        # x_add = x_1 + self.conv72(x)
        # x = nn.functional.relu(x_add)  # [8 32 4]

        # x = nn.functional.relu(self.conv8(x))  # [8 4 1]
        #        x = x.view(-1, 64*1*1)
        x = x.reshape(-1, 32 * 4)
        c0 = tc.abs(self.fc_c(x))#.clamp_(clamp_min, clamp_max)
        d0 = tc.abs(self.fc_d(x))#.clamp_(clamp_min, clamp_max)
        T0 = tc.abs(self.fc_T(x))#.clamp_(clamp_min, clamp_max)
        # x = tc.abs(self.fc1(x)).clamp_(clamp_min, clamp_max)  # [batch 58]
        return c0, d0, T0

class HyperNetOptimized(nn.Module):
    def __init__(self, kernel=3, pad=1, layer_num=20):
        super(HyperNetOptimized, self).__init__()

        # 初始升维
        self.conv1 = nn.Conv1d(1, 32, kernel_size=kernel, padding=pad)

        # Multi-Scale Feature Extractor (CPMB1)
        self.branch_k3 = nn.Conv1d(32, 16, kernel_size=3, padding=1, stride=2)
        self.branch_k5 = nn.Conv1d(32, 16, kernel_size=5, padding=2, stride=2)
        self.branch_k7 = nn.Conv1d(32, 16, kernel_size=7, padding=3, stride=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # Output: [B, 48, L/4]

        # CPMB2
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3, padding=1, stride=2)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=2)
        self.pool2 = nn.AdaptiveAvgPool1d(4)  # Output: [B, 64, 4]

        # Attention module (optional)
        self.se_fc1 = nn.Linear(64, 16)
        self.se_fc2 = nn.Linear(16, 64)

        # Fully connected layers for parameters
        self.fc_c = nn.Linear(64 * 4, layer_num - 1)
        self.fc_d = nn.Linear(64 * 4, layer_num - 1)
        self.fc_T = nn.Linear(64 * 4, layer_num)

    def forward(self, x, clamp):
        x = self.conv1(x)  # [B, 32, L]

        # CPMB1 多尺度感知
        x_k3 = F.relu(self.branch_k3(x))
        x_k5 = F.relu(self.branch_k5(x))
        x_k7 = F.relu(self.branch_k7(x))
        x = torch.cat([x_k3, x_k5, x_k7], dim=1)  # [B, 48, L/2]
        x = self.pool1(x)  # [B, 48, L/4]

        # CPMB2
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool2(x)  # [B, 64, 4]

        # Squeeze-and-Excitation Attention
        se = torch.mean(x, dim=2)  # [B, 64]
        se = F.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se)).unsqueeze(-1)  # [B, 64, 1]
        x = x * se

        # Flatten and predict
        x = x.view(x.size(0), -1)
        c0 = torch.abs(self.fc_c(x)).clamp(clamp[0], clamp[1])
        d0 = torch.abs(self.fc_d(x))  # 可选加 clamp
        T0 = torch.abs(self.fc_T(x)).clamp(clamp[0], clamp[1])
        return c0, d0, T0


# 修改 network/cfmf_cdT.py 中的 HyperNet 定义
class HyperNet_Context2D(nn.Module):
    def __init__(self, layer_num, in_channels=2):
        """
        利用 Batch 维度作为上下文信息的超网络
        in_channels: 2 (实部 + 虚部)
        """
        super(HyperNet_Context2D, self).__init__()
        
        # 1. 2D 卷积层：同时提取 Range (H) 和 Time (W/Batch) 特征
        # 输入形状虚拟为: [1, 2, Range(256), Batch(Time)]
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        
        # 2. 池化层：我们只压缩 Range 维度，保留 Time(Batch) 维度
        # AdaptiveAvgPool2d((1, None)) 表示 H 变为 1，W (Batch) 保持原样
        self.pool = nn.AdaptiveAvgPool2d((1, None)) 
        
        # 3. 参数预测头 (使用 1D 卷积在时间轴上进一步融合)
        # 输入: [1, 64, Batch]
        self.head_c = nn.Conv1d(64, layer_num - 1, kernel_size=3, padding=1)
        self.head_d = nn.Conv1d(64, layer_num - 1, kernel_size=3, padding=1)
        self.head_T = nn.Conv1d(64, layer_num, kernel_size=3, padding=1)

    def forward(self, x, clamp_min, clamp_max):
        """
        x: [Batch, 2, 256] (Real, Imag)
        """
        # 1. 维度变换：将 Batch 移到最后作为“宽度/时间”
        # [Batch, 2, 256] -> [2, 256, Batch] -> [1, 2, 256, Batch]
        # 这里 1 是虚拟的 batch_size (因为我们把真正的 batch 当作了图的宽度)
        x_img = x.permute(1, 2, 0).unsqueeze(0)
        
        # 2. 提取时空特征
        feat = F.relu(self.conv1(x_img))
        feat = F.relu(self.conv2(feat)) # [1, 64, 256, Batch]
        
        # 3. 压缩 Range 维度，保留 Batch 维度
        feat = self.pool(feat) # [1, 64, 1, Batch]
        feat = feat.squeeze(2) # [1, 64, Batch]
        
        # 4. 预测参数
        c = torch.abs(self.head_c(feat)) # [1, Layers-1, Batch]
        d = torch.abs(self.head_d(feat))
        T = torch.abs(self.head_T(feat))
        
        # 5. 还原维度以匹配 SBL 的输入要求
        # [1, L, B] -> [B, L]
        c = c.squeeze(0).permute(1, 0) # [Batch, Layers-1]
        d = d.squeeze(0).permute(1, 0)
        T = T.squeeze(0).permute(1, 0)
        
        # 限制范围
        c = c.clamp(min=clamp_min, max=clamp_max)
        d = d.clamp(min=clamp_min) # d 通常不需要上限
        T = T.clamp(min=clamp_min, max=clamp_max)
        
        return c, d, T
    
class cfmf(nn.Module):
    def __init__(self, Para):
        super(cfmf, self).__init__()

        self.Nlayer = Para['layer_num']
        self.P = Para['P']  # 256
        self.Q = Para['Q']  # 1
        self.N = Para['N']

        # a0, b0 作为可学习参数
        self.a0 = nn.Parameter(tc.tensor(Para['a0']))
        self.b0 = nn.Parameter(Para['b0'] * tc.ones(1))

        # c0, d0, T 由 HyperNet 生成，不再作为 nn.Parameter
        # self.h = HyperNet_7convL_CU_5_1(3, 1, self.Nlayer)  # HyperNet_1   HyperNet_7convL_CU_5_1
        self.h = HyperNet_Context2D(layer_num=self.Nlayer, in_channels=2)
    
    def forward(self, Y, zidian, clamp_min, clamp_max):
        A = zidian['A']
        AT = zidian['AT']
        AA = zidian['AA']
        M = A.shape[1]  # 65536
        N = Y.shape[2]  # batch 1 256 1
        # 256 256
        batch_size = Y.shape[0]

        AA = AA.unsqueeze(0)  # [1,128,128]
        AY = tc.matmul(AT, Y)  # 65536 1

        # c, d, T = self.h(abs(AY).float().squeeze(3), clamp_min, clamp_max)  # [batch 58]
        # === 修改开始 ===
        # 1. 先把数据展平为 [Batch, 256]
        # 使用 reshape(batch_size, -1) 是最安全的写法，能自动处理中间多余的 1 维度
        ay_real_flat = AY.real.reshape(batch_size, -1) # [Batch, 256]
        ay_imag_flat = AY.imag.reshape(batch_size, -1) # [Batch, 256]
        # 2. 增加通道维度 -> [Batch, 1, 256]
        ay_real = ay_real_flat.unsqueeze(1)
        ay_imag = ay_imag_flat.unsqueeze(1)
        
        # 拼接: [Batch, 2, 256]
        ay_input = torch.cat([ay_real, ay_imag], dim=1) 

        # 2. 调用 Context HyperNet
        # 注意：这里不需要再 .float()，因为 real/imag 取出来已经是 float 了
        c, d, T = self.h(ay_input, clamp_min, clamp_max)
        # === 修改结束 ===
        
        c = c.reshape(batch_size, -1, 1, 1)  # [batch 3N-2 1 1]
        d = d.reshape(batch_size, -1, 1, 1)
        T = T.reshape(batch_size, -1, 1, 1)

        # 修改前
        # U = tc.zeros([batch_size, 1, M, 1], dtype=tc.complex128).cuda()

        # 修改后 (明确 complex64)
        U = tc.zeros([batch_size, 1, M, 1], dtype=tc.complex64).cuda()
        eps = (self.a0 / self.b0) * tc.ones(batch_size, 1, 1, 1).cuda() #(默认float32，这行通常没事，但为了保险)
        # eps 保持默认即可，因为它与 complex64 相乘会自动适配，但要确保它不是 double

        Lamda = c[:, 0, :, :].unsqueeze(-1) / (d[:, 0, :, :].unsqueeze(-1) * tc.ones(batch_size, 1, M, 1).cuda())

        for k_layer in range(self.Nlayer - 1):
            _c0 = c[:, k_layer, :, :].unsqueeze(-1)  # 19,1,1,1
            _d0 = d[:, k_layer, :, :].unsqueeze(-1)  # 19,1,1,1
            _T = T[:, k_layer, :, :].unsqueeze(-1)  # 20,1,1,1
            U, eps, Lamda, c0, d0 = self.layer(Y, zidian, AA, AY, U, eps, Lamda, _T,
                                               self.a0, self.b0[0], _c0, _d0, N)
            # U, eps, Lamda = self.layer(Y, zidian, AA, AY, U, eps, Lamda, self.T[k_layer],
            #                            self.a0, self.b0[0], self.c0[k_layer], self.d0[k_layer], N)

        _T = T[:, self.Nlayer - 1, :, :].unsqueeze(-1)
        U = self.last_layer(Y, AA, AY, U, eps, Lamda, _T)

        
        c = tc.squeeze(c.detach().cpu())[batch_size - 1, :].reshape(1, -1)
        d = tc.squeeze(d.detach().cpu())[batch_size - 1, :].reshape(1, -1)
        T = tc.squeeze(T.detach().cpu())[batch_size - 1, :].reshape(1, -1)

        return U, c, d, T

    def layer(self, Y, zidian, AA, AY, U, eps, Lamda, T, a0, b0, c0, d0, N):
        # Update X
        batch_size = Y.shape[0]
        Delta = U
        M = U.shape[2]
        U1 = T * Delta  # 20 1 65536 1
        U2 = AA @ Delta  # 20 1 65536 1
        U3 = AY  # 65536 1
        c = c0 + 1
        a = a0 + N
        temp1 = tc.div(tc.ones(batch_size, 1, M, 1).cuda(), (eps * T + Lamda+1e-6))  # 20 1 N2 1
        U = eps * (U1 - U2 + U3) * temp1  # 20 1 N2 1

        D = myDiag(AA).unsqueeze(2).cuda()  # 1 N2 1
        # 修改点 2: Sigma 分母增加 1e-6
        # 原始: temp = eps * D + Lamda
        # 原始: Sigma = tc.div(tc.ones(batch_size, 1, M, 1).cuda(), temp)
        temp = eps * D + Lamda + 1e-6
        Sigma = tc.div(tc.ones(batch_size, 1, M, 1).cuda(), temp)

        # Update eps --- precision of Noise
        A = zidian['A']

        eps1 = tc.norm(Y - A @ U, 'fro', dim=[2, 3], keepdim=True) ** 2
        eps2 = tc.sum(D * Sigma, dim=[2, 3], keepdim=True)
        
        # 修改点 3: eps 更新分母增加 1e-6  # *：矩阵元素点乘；@矩阵相乘
        # eps = a / (b0 + (eps1 + eps2))  # size N_size*1
        eps = a / (b0 + (eps1 + eps2) + 1e-6)

        # Update Lamda
        # 修改点 4: Lamda 更新分母增加 1e-6
        # 原始: Lamda = c / (d0 + (tc.abs(U) ** 2 + Sigma))
        Lamda = c / (d0 + (tc.abs(U) ** 2 + Sigma) + 1e-6)

        return U, eps, Lamda, c0, d0

    def last_layer(self, Y, AA, AY, U, eps, Lamda, T):
        batch_size = Y.shape[0]
        Delta = U
        U1 = T * Delta
        U2 = AA @ Delta
        U3 = AY
        P = U.shape[2]
        N = self.N

        # 修改点 1: temp1 分母增加 1e-6
        # 原始: temp1 = tc.div(tc.ones(batch_size, 1, P, 1).cuda(), (eps * T + Lamda))
        temp1 = tc.div(tc.ones(batch_size, 1, P, 1).cuda(), (eps * T + Lamda + 1e-6))
        
        U = eps * (U1 - U2 + U3) * temp1  # size: N_size*P*Q
        # U = tc.reshape(U, (-1, 1, N, N)).transpose(3, 2)
        # U1 = U.detach().data.cpu().numpy()
        # U1 = np.fft.ifftshift(U1, 0)
        # U2 = tc.from_numpy(U1).cuda()

        # 修改点 2: 归一化分母防止除零
        # 原始: U = U / tc.max(abs(U))
        U = U / (tc.max(abs(U)) + 1e-8)
        return abs(U)


def myDiag(x):
    out = tc.zeros(x.shape[0], x.shape[1])
    for i in range(out.shape[0]):
        out[i] = x[i].diag()
    return out
