import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from torchvision import transforms
from torch.utils.data import DataLoader
# import seaborn as sns
import torch
import torch.nn as nn
from network.classifier import CNN5,CNN5_Improved,CustomDataset,CNN5_1
import scipy.io as sio

def load_saved_patches(data_dir,epoch):
    checkfile = os.path.join(data_dir,f'train_patch_{epoch}.pt')
    data = torch.load(checkfile)
    final_outputs = data['image']
    final_labels = data['label']
    return final_outputs, final_labels

output_dir = "./output_data/"
num_epochs=50
batch_size=10
learning_rate=1e-3


# 加载保存的结果
image, label = load_saved_patches(output_dir,100)

if isinstance(image, torch.Tensor):
    image_np = image.cpu().numpy()
else:
    image_np = image

mat_save_path = os.path.join(output_dir, "image.mat")
sio.savemat(mat_save_path, {'image': image_np})
print(f"image 已保存为: {mat_save_path}")



model = CNN5_1(num_classes=3)
model_dir = "./weights/weight5_13/"
# 直接将网络输出输入到分类器
# 创建数据集
dataset = CustomDataset(image, label)
print(f"数据集大小：{len(dataset)}")

# 使用整个数据集作为训练集
train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

# 初始化模型

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 训练设置
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=2e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)  # 学习率衰减

# 初始化记录
train_losses = []
train_accuracies = []

# 训练分类模型
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0
    correct = 0
    train_preds = []
    train_true = []
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)  # 获取预测结果
        train_preds.extend(preds.cpu().numpy())
        train_true.extend(labels.cpu().numpy())

        correct += (preds == labels).sum().item()
    
    # scheduler.step()  # 更新学习率
    train_acc = accuracy_score(train_true, train_preds)
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(train_acc)
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Train Acc: {train_acc:.4f}")
    
    # 保存模型
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, f'model-{epoch+1}.pth'))

# 绘制损失曲线
save_dir = "./output_image"
os.makedirs(save_dir, exist_ok=True)

plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
loss_curve_path = os.path.join(save_dir, "loss_curve.png")
plt.savefig(loss_curve_path)
print(f"损失曲线已保存到: {loss_curve_path}")
plt.close()  # 释放资源

plt.figure()
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
accuracy_curve_path = os.path.join(save_dir, "accuracy_curve.png")
plt.savefig(accuracy_curve_path)
print(f"准确率曲线已保存到: {accuracy_curve_path}")
plt.close()  # 释放资源