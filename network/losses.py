import torch
import torch.nn as nn

class PearsonLoss(nn.Module):
    """
    Pearson correlation coefficient loss function.
    
    Args:
        weight_pearson (float): Weight for the Pearson loss term
        weight_mse (float): Weight for the MSE loss term
    """
    def __init__(self, weight_pearson=65.0, weight_mse=0.4):
        super(PearsonLoss, self).__init__()
        self.weight_pearson = weight_pearson
        self.weight_mse = weight_mse
        
    def forward(self, pred, target):
        """
        Calculate the combined loss using Pearson correlation and MSE.
        
        Args:
            pred (torch.Tensor): Predicted values
            target (torch.Tensor): Target values
            
        Returns:
            torch.Tensor: Combined loss value
        """
        # Calculate MSE loss
        mse_loss = torch.sum((torch.abs(pred) - torch.abs(target)) ** 2)
        
        # Calculate Pearson correlation
        X = pred.squeeze()  # 预测结果
        I = torch.abs(target).squeeze()  # 标签
        
        # Calculate means
        mex = torch.mean(X, 1).view(X.size(0), 1)
        mei = torch.mean(I, 1).view(I.size(0), 1)
        
        # Calculate standard deviations
        sigmax = torch.sqrt(torch.mean((X - mex) ** 2))
        sigmai = torch.sqrt(torch.mean((I - mei) ** 2))
        
        # Calculate covariance
        convv = torch.mean(torch.abs(X - mex) * torch.abs(I - mei))
        sigmul = sigmax * sigmai
        
        # Calculate Pearson correlation
        pearson = convv / sigmul
        
        # Calculate final loss
        loss = (1 - pearson) * self.weight_pearson + mse_loss * self.weight_mse
        
        return loss, pearson.item(), mse_loss.item()
    
    def __repr__(self):
        return f'PearsonLoss(weight_pearson={self.weight_pearson}, weight_mse={self.weight_mse})'


# 修改前
# criterion = PearsonLoss(weight_pearson=ps_list[0], weight_mse=ps_list[1])

# 修改后 (引入 L1 Loss 组合)
# 1. 定义组合损失函数类 (可以直接写在 Train.py 里，或者只是简单的 lambda)
import torch
import torch.nn as nn
import torch
import torch.nn as nn


class CompositeLoss(nn.Module):
    def __init__(self, weight_pearson=65.0, weight_mse=0.4, weight_l1=0.1, weight_smooth=0.5): # 建议平滑权重 0.1 ~ 0.5
        super(CompositeLoss, self).__init__()
        self.weight_pearson = weight_pearson
        self.weight_mse = weight_mse
        self.weight_l1 = weight_l1
        self.weight_smooth = weight_smooth # 新增平滑权重
        
    def forward(self, pred, target):
        """
        pred: [Batch, 1, 256, 1] (复数或实数幅度)
        """
        pred_mag = torch.abs(pred)
        target_mag = torch.abs(target)

        # 1. MSE Loss
        mse_loss = torch.sum((pred_mag - target_mag) ** 2)
        
        # 2. L1 Loss (控制线条变细)
        l1_loss = torch.sum(pred_mag) # 直接对预测幅度求和，强迫稀疏

        # 3. Pearson Loss (控制波形相似)
        X = pred_mag.view(pred_mag.shape[0], -1)
        I = target_mag.view(target_mag.shape[0], -1)
        
        # ... (Pearson 计算保持原样，略) ...
        # 为了简洁，这里用伪代码表示 Pearson 计算过程
        # (请保留你原来的 Pearson 代码)
        mex = torch.mean(X, 1, keepdim=True)
        mei = torch.mean(I, 1, keepdim=True)
        sigmax = torch.sqrt(torch.mean((X - mex) ** 2, dim=1, keepdim=True))
        sigmai = torch.sqrt(torch.mean((I - mei) ** 2, dim=1, keepdim=True))
        convv = torch.mean((X - mex) * (I - mei), dim=1, keepdim=True)
        pearson = torch.mean(convv / (sigmax * sigmai + 1e-6))

        # === 4. 新增：时序平滑损失 (Temporal Smoothness Loss) ===
        # 计算 pred[t] 和 pred[t-1] 的 L1 距离
        # 假设 Batch 是连续的，我们惩罚相邻两帧的突变
        if pred_mag.shape[0] > 1:
            # diff = |Frame_t - Frame_{t-1}|
            diff = torch.abs(pred_mag[1:] - pred_mag[:-1])
            # 我们希望差异越小越好（连贯），但不要为0（允许缓慢变化）
            smooth_loss = torch.sum(diff) 
        else:
            smooth_loss = torch.tensor(0.0).to(pred.device)

        # 总损失
        # 重点：适当加大 weight_smooth 可以让线条更连贯，但太大了一根线会拖成一片
        loss = (1 - pearson) * self.weight_pearson + \
               mse_loss * self.weight_mse + \
               l1_loss * self.weight_l1 + \
               smooth_loss * self.weight_smooth
        
        return loss, pearson.item(), mse_loss.item(), l1_loss.item(), smooth_loss.item()