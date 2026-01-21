import os
import numpy as np
import matplotlib
import torch
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
plt.ioff()  # 禁用交互模式

def dispimage(C, db=20, r=None, xr=None, cmap='jet', cmax=None, save_path=None, title='STFT Result'):
    """
    显示二维时频图像，支持动态范围、颜色图、坐标轴控制等。
    """
    C = np.array(C)
    if r is None:
        r = np.arange(C.shape[1])
    if xr is None:
        xr = np.arange(C.shape[0])
    if cmax is None:
        cmax = np.max(np.abs(C))

    dynrng = abs(db)
    ra = cmax / (10 ** (dynrng / 20))
    C_show = np.where(np.abs(C) >= ra, C, ra)

    if db > 0:
        img = 20 * np.log10(np.abs(C_show) / cmax + 1e-12)
    else:
        img = 20 * np.abs(C_show)

    # 创建新的图形
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(img, extent=[r[0], r[-1], xr[-1], xr[0]], aspect='auto', cmap=cmap)
    
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',pad_inches=0)
        print(f"图像已保存: {save_path}")
    
    # 清理当前图形
    plt.close(fig)  # 仅关闭当前图形

def convert_images_pt_to_png(final_outputs, output_dir):
    """
    将保存的 .pt 文件中的 images 批量转为 png 图片并保存到 output_dir。
    :param pt_file: .pt 文件路径
    :param output_dir: 图片输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 加载 images
    all_images = final_outputs
    img_count = 0
    for idx, images in enumerate(all_images):
        img_np = images.squeeze() # 先detach再numpy
        save_path = os.path.join(
            output_dir,
            f"img{idx+1}_{img_count}.png"
        )
        img_np = np.fft.fftshift(img_np, axes=1)
        img_np = np.rot90(img_np, k=1,axes=(0, 1))  # 逆时针旋转90度
        dispimage(img_np, save_path=save_path, title=f'img{idx+1}_{img_count}.png')
        img_count += 1
    print(f"已保存 {img_count} 张图片到 {output_dir}")

def load_saved_patches(data_dir):
    # checkfile = os.path.join(data_dir,f'train_patch_20.pt')
    checkfile = os.path.join(data_dir,f'train_patch_50.pt')
    data = torch.load(checkfile)
    final_outputs = data['image']
    final_labels = data['label']
    return final_outputs, final_labels

if __name__ == "__main__":
    # pt_file = r"D:\新建文件夹\te\img_output\img5_9\images_epoch_100.pt"  # 修改为你的.pt文件路径
    # output_dir = r"D:\新建文件夹\te\img_output\img5_9\epoch100_png"      # 修改为你的输出目录
    # convert_images_pt_to_png(pt_file, output_dir)
    
    outputimg_dir = os.path.join("./img_output/train_img/")  #outputimg_dir = os.path.join("./img_output/train_img12_30/")
    output_dir = "./output_data/"
    final_outputs, final_labels = load_saved_patches(output_dir)
    convert_images_pt_to_png(final_outputs,outputimg_dir)
