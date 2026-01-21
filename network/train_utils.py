import torch
import numpy as np
import time
import gc
import os
import json
from scipy.io import savemat
from network import util

def load_checkpoint(model_dir, net, optimizer=None, epoch=None):
    """
    加载检查点
    参数:
        model_dir: 模型目录
        net: 模型
        optimizer: 优化器
        epoch: 指定加载的epoch，如果为None则加载最新的
    返回:
        start_epoch: 开始的epoch
        best_loss: 最佳损失
    """
    if epoch is None:
        # 找到最新的检查点
        checkpoints = [f for f in os.listdir(model_dir) if f.startswith('checkpoint_epoch_')]
        if not checkpoints:
            return 0, float('inf')
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_path = os.path.join(model_dir, latest_checkpoint)
    else:
        checkpoint_path = os.path.join(model_dir, f'checkpoint_epoch_{epoch}.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"指定的检查点文件不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['loss']
    
    return start_epoch, best_loss

def save_outputs(indices, idx, x_out, x_te, output_dir):
    """
    保存输出和标签为 .mat 文件。
    参数:
        indices: 数据索引列表
        x_out: 模型输出
        x_te: 标签数据
        output_dir: 输出目录
    """
    for i in range(len(indices)):
        idx = indices[i].item()
        output = x_out[i]
        output_numpy = np.expand_dims(output, 0)
        
    image = [] 
    label = []
    for i in range(len(indices)):
        start_idx = int(i*256)
        end_idx = int((i+1)*256)
        idx = indices[start_idx:end_idx].item()
        output_numpy = x_out[start_idx:end_idx].detach().cpu().numpy()
        image.append(torch.tensor(output_numpy))
        label.append(idx)
    return image, label       
     

def train_model(net, data_loader, optimizer, criterion, args, logger, num_iter_epoch, zidiancuda, clamp_list, start_epoch=0, phase='train'):
    """
    训练模型的主循环
    参数:
        net: 模型
        data_loader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        args: 参数
        logger: 日志记录器
        num_iter_epoch: 每个epoch的迭代次数
        zidiancuda: 字典数据
        clamp_list: 裁剪范围
        start_epoch: 开始训练的epoch
        start_phase: 开始训练的阶段 ('train', 'visualization', 'classification')
    返回:
        Loss: 训练损失记录
    """
    Loss = torch.zeros(args.epochs)
    total_time = 0
    
    # 用于收集最后一个epoch的输出
    final_outputs = []
    final_labels = []
    
    # 训练阶段
    for epoch in range(start_epoch, args.epochs):
        # 训练阶段
        net.train()
        loss_per_epoch = 0
        mse_per_epoch = 0
        pearson_per_epoch = 0
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
            loss.backward()
            optimizer.step()
            
            # 更新统计
            loss_per_epoch += loss.item() / num_iter_epoch[phase]
            ssim_iter = util.batch_SSIM(abs(x_out), abs(x_te))
            pearson_per_epoch += pearson_iter / num_iter_epoch[phase]
            mse_per_epoch += mse_iter/ num_iter_epoch[phase] 
            
            # 打印进度
            if (ii + 1) % np.floor(num_iter_epoch[phase] / 2) == 0:
                logger.info(
                    '------> [Epoch:{:>2d}/{:<2d}] {:s}:{:0>4d}/{:0>4d}, Loss={:+.2e},Pearson={:+.2e}, lr={:.1e}'.format(
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
        
        # 计算epoch耗时
        epoch_time = time.time() - epoch_start_time
        total_time += epoch_time
        
        # 记录epoch结果
        logger.info(
            '------> [Epoch:{:>2d}/{:<2d}] {:s}: Loss={:+.4f} {:+.4f} {:+.4f}, Time={:.2f}s'.format(
                epoch + 1,
                args.epochs,
                phase,
                loss_per_epoch,
                pearson_per_epoch,
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
        
        # 保存检查点（每10轮保存一次，文件名包含轮次和皮尔逊指标）
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.model_dir, f'model_epoch_{epoch + 1}_pearson_{pearson_per_epoch:.4f}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'pearson': pearson_per_epoch,
                'loss': loss_per_epoch
            }, checkpoint_path)


        # 清理内存
        gc.collect()
        torch.cuda.empty_cache()

    # 打印总训练时间
    logger.info(f'总训练时间: {total_time:.2f}秒')
    logger.info(f'平均每个epoch时间: {total_time/args.epochs:.2f}秒')

    return Loss, final_outputs, final_labels

def train_model2(net, data_loader, optimizer, scheduler ,criterion, args, logger, num_iter_epoch, zidiancuda, clamp_list, start_epoch=0, phase='train'):
    """
    训练模型的主循环
    参数:
        net: 模型
        data_loader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        args: 参数
        logger: 日志记录器
        num_iter_epoch: 每个epoch的迭代次数
        zidiancuda: 字典数据
        clamp_list: 裁剪范围
        start_epoch: 开始训练的epoch
        start_phase: 开始训练的阶段 ('train', 'visualization', 'classification')
    返回:
        Loss: 训练损失记录
    """
    Loss = torch.zeros(args.epochs)
    total_time = 0
    
    # 用于收集最后一个epoch的输出
    final_outputs = []
    final_labels = []
    
    # 训练阶段
    for epoch in range(start_epoch, args.epochs):
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
            loss, pearson_iter, mse_iter, l1_iter,smooth_loss = criterion(x_out, x_te)
            ssim_iter = util.batch_SSIM(abs(x_out), abs(x_te))
            loss.backward()
            optimizer.step()
            
            # 更新统计
            loss_per_epoch += loss.item() / num_iter_epoch[phase]
            ssim_per_epoch += ssim_iter / num_iter_epoch[phase]
            pearson_per_epoch += pearson_iter / num_iter_epoch[phase]
            mse_per_epoch += mse_iter/ num_iter_epoch[phase] 
            l1_per_epoch = l1_iter / num_iter_epoch[phase]
            smooth_per_epoch = smooth_loss / num_iter_epoch[phase]
            
            # 打印进度
            if (ii + 1) % np.floor(num_iter_epoch[phase] / 2) == 0:
                logger.info(
                    '------> [Epoch:{:>2d}/{:<2d}] {:s}:{:0>4d}/{:0>4d}, Loss={:+.2e},Pearson={:+.4e}, l1_iter={:+.4f},smooth={:+.4f},lr={:.1e}'.format(
                        epoch + 1,
                        args.epochs,
                        phase,
                        ii + 1,
                        num_iter_epoch[phase],
                        loss_per_epoch,
                        pearson_per_epoch,
                        l1_per_epoch,
                        smooth_per_epoch,
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

        # 保存检查点（每10轮保存一次，文件名包含轮次和皮尔逊指标）
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.model_dir, f'model_epoch_{epoch + 1}_pearson_{pearson_per_epoch:.4f}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'pearson': pearson_per_epoch,
                'loss': loss_per_epoch
            }, checkpoint_path)


        # 清理内存
        gc.collect()
        torch.cuda.empty_cache()

    # 打印总训练时间
    logger.info(f'总训练时间: {total_time:.2f}秒')
    logger.info(f'平均每个epoch时间: {total_time/args.epochs:.2f}秒')

    return Loss, final_outputs, final_labels

def test_model(net, data_loader, criterion, args, num_iter_epoch, zidiancuda, clamp_list,phase='test'):
    """
    训练模型的主循环
    参数:
        net: 模型
        data_loader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        args: 参数
        logger: 日志记录器
        num_iter_epoch: 每个epoch的迭代次数
        zidiancuda: 字典数据
        clamp_list: 裁剪范围
        start_epoch: 开始训练的epoch
        start_phase: 开始训练的阶段 ('train', 'visualization', 'classification')
    返回:
        Loss: 训练损失记录
    """
    Loss = torch.zeros(args.epochs)
    total_time = 0
    
    # 用于收集最后一个epoch的输出
    final_outputs = []
    final_labels = []
    
    net.eval()
    loss_per_epoch = 0
    ssim_per_epoch = 0
    pearson_per_epoch = 0
    num_data = len(data_loader[phase].dataset)
    # 记录epoch开始时间
    start_time = time.time()
    
    output_numpy = []
    idx_all = []
    for ii, data in enumerate(data_loader[phase]):
        y_te, x_te, indices,idx = data
        y_te = y_te.cuda()
        x_te = x_te.cuda()
        with torch.no_grad():
            x_out, c0, d0, T = net(y_te, zidiancuda, clamp_list[0], clamp_list[1])
        
        # 保存最后一个epoch的输出
        
        # if (epoch + 1) % args.save_model_freq == 0 or epoch + 1 == args.epochs:
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

        # 计算损失
        loss, pearson_iter, mse_iter, l1_iter = criterion(x_out, x_te)
        
        # 更新统计
        loss_per_epoch += loss.item() / num_iter_epoch[phase]
        ssim_iter = util.batch_SSIM(abs(x_out), abs(x_te))
        pearson_per_epoch += pearson_iter / num_iter_epoch[phase]
        l1_per_epoch = l1_iter / num_iter_epoch[phase]
        
    
    # 计算epoch耗时
    total_time = time.time() - start_time
    
    # 打印模型参数
    print(c0)
    print(d0)
    print(T)
    
    
    # 保存损失     
    Loss = loss_per_epoch

    # 清理内存
    gc.collect()
    torch.cuda.empty_cache()
    
    # 打印总训练时间
    print(f'总训练时间: {total_time:.2f}秒')
    print(f'平均每个epoch时间: {total_time/args.epochs:.2f}秒')
    
    # 将收集的输出转换为张量
    if image is not None:
        final_labels = torch.tensor(label)
        final_outputs = torch.stack(image)
        final_outputs = final_outputs.squeeze(-1)

    save_path = os.path.join(args.output_dir, f'test_patch.pt')
    torch.save({'image': final_outputs,
                'label': final_labels
            }, save_path)
 
    return Loss, final_outputs, final_labels

def clean_memory():
    """清理GPU内存"""
    gc.collect()
    torch.cuda.empty_cache()

def count_param(model):
    """计算模型参数量"""
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count