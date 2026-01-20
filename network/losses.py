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

class CompositeLoss(nn.Module):
    """
    结合 Pearson Loss, MSE (Sum of Squared Error) 和 L1 Loss 的组合损失函数。
    """
    def __init__(self, weight_pearson=65.0, weight_mse=0.4, weight_l1=0.1):
        super(CompositeLoss, self).__init__()
        self.weight_pearson = weight_pearson
        self.weight_mse = weight_mse
        self.weight_l1 = weight_l1
        
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): 预测值 (复数或实数)
            target (torch.Tensor): 标签值
        """
        # 1. 预处理：取幅度 (SBL 重构通常关注幅度谱的稀疏性)
        pred_mag = torch.abs(pred)
        target_mag = torch.abs(target)

        # 2. 计算 MSE Loss (保持您原有的 Sum 逻辑，而非 Mean)
        mse_loss = torch.sum((pred_mag - target_mag) ** 2)
        
        # 3. 计算 L1 Loss (稀疏约束的关键)
        # 使用 sum 还是 mean 取决于您的权重。为了与您的 mse_loss (sum) 量级匹配，这里也用 sum
        l1_loss = torch.sum(torch.abs(pred_mag - target_mag))

        # 4. 计算 Pearson Correlation (保持原有逻辑)
        X = pred.squeeze()  
        I = torch.abs(target).squeeze()
        
        # 维度处理，防止 batch 为 1 时 squeeze 掉 batch 维
        if X.ndim == 1: X = X.unsqueeze(0)
        if I.ndim == 1: I = I.unsqueeze(0)

        mex = torch.mean(X, 1, keepdim=True)
        mei = torch.mean(I, 1, keepdim=True)
        
        sigmax = torch.sqrt(torch.mean((X - mex) ** 2, dim=1, keepdim=True))
        sigmai = torch.sqrt(torch.mean((I - mei) ** 2, dim=1, keepdim=True))
        
        # 加上 1e-8 防止除以零 (单精度下非常重要)
        convv = torch.mean(torch.abs(X - mex) * torch.abs(I - mei), dim=1, keepdim=True)
        sigmul = sigmax * sigmai + 1e-8
        
        pearson = torch.mean(convv / sigmul)
        
        # 5. 总损失
        loss = (1 - pearson) * self.weight_pearson + mse_loss * self.weight_mse + l1_loss * self.weight_l1
        
        # 返回4个值，以便 train_utils 记录日志
        return loss, pearson.item(), mse_loss.item(), l1_loss.item()

    def __repr__(self):
        return f'CompositeLoss(w_p={self.weight_pearson}, w_m={self.weight_mse}, w_l1={self.weight_l1})'