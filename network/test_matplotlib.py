import os
import numpy as np
from scipy.io import savemat, loadmat
from visualization import visualize_stft_outputs

# 获取脚本所在目录的绝对路径
base_dir = os.path.dirname(os.path.abspath(__file__))  # 动态获取脚本所在目录
output_dir = os.path.join(base_dir, "test_outputs")  # 使用 os.path.join 生成路径
save_dir = os.path.join(base_dir, "test_images")     # 使用 os.path.join 生成路径

# 动态创建目录
os.makedirs(output_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

# 生成测试 .mat 文件
print("正在生成测试 .mat 文件...")
for i in range(5):  # 创建 5 个测试文件
    data = {'stft': np.random.rand(256, 256)}  # 随机生成 256x256 的矩阵
    savemat(os.path.join(output_dir, f'out_{i}.mat'), data)
print(f"测试 .mat 文件已生成，保存在: {output_dir}")

# 检查生成的 .mat 文件内容
print("检查生成的 .mat 文件内容...")
for i in range(5):
    mat_file = os.path.join(output_dir, f'out_{i}.mat')
    data = loadmat(mat_file)
    print(f"文件 {mat_file} 内容: {list(data.keys())}")
    if 'stft' in data:
        print(f"'stft' 数据形状: {data['stft'].shape}")
    else:
        print(f"文件 {mat_file} 中没有找到 'stft' 数据")


# 调用 visualize_stft_outputs
print("开始测试 visualize_stft_outputs 函数...")
visualize_stft_outputs(output_dir, save_dir)
print(f"测试完成，图像已保存到: {save_dir}")