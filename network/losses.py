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
class CompositeLoss(torch.nn.Module):
    def __init__(self, pearson_weight, mse_weight, l1_weight=0.05):
        super().__init__()
        self.pearson = PearsonLoss(weight_pearson=pearson_weight, weight_mse=mse_weight)
        self.l1 = torch.nn.L1Loss()
        self.l1_weight = l1_weight

    def forward(self, outputs, labels):
        # Pearson + MSE 保持原样
        loss_p = self.pearson(outputs, labels)
        # 增加 L1 稀疏约束 (假设 labels 是纯净信号，如果不纯可以用 torch.zeros_like(outputs))
        loss_l1 = self.l1(outputs, labels) 
        return loss_p + self.l1_weight * loss_l1

