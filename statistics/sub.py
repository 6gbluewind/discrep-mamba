import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from monai import transforms as tfs
from scipy.ndimage import sobel
from utils import load_and_transform, get_transform, create_logger

matplotlib.use('Agg')

log_dir = './output_contour'
os.makedirs(log_dir, exist_ok=True)
logger = create_logger(logging_dir=log_dir)


def compute_gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """计算二维图像的梯度幅度（Sobel算子）"""
    grad_x = sobel(img, axis=1)  # 水平方向梯度
    grad_y = sobel(img, axis=0)  # 垂直方向梯度
    return np.sqrt(grad_x ** 2 + grad_y ** 2)


def plot_modality_differences(sample: dict,
                              mod1: str,
                              mod2: str,
                              save_path: str = None):
    """
    绘制两个模态中间切片的强度差和梯度差图像

    Args:
        sample: 包含各模态numpy数组的字典，数组形状为 (1, H, W, D)
        mod1: 第一个模态名称
        mod2: 第二个模态名称
        save_path: 保存路径，若为None则显示图像
    """
    # 获取数据（去掉batch维度）
    arr1 = sample[mod1][0]  # shape: (H, W, D)
    arr2 = sample[mod2][0]

    # 中间切片索引（深度维为最后一个轴）
    mid_slice = arr1.shape[-1] // 2
    slice1 = arr1[:, :, mid_slice]
    slice2 = arr2[:, :, mid_slice]

    # 计算强度差
    diff_intensity = slice1 - slice2

    # 计算梯度幅度差
    grad1 = compute_gradient_magnitude(slice1)
    grad2 = compute_gradient_magnitude(slice2)
    diff_grad = grad1 - grad2

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 强度差图
    im1 = axes[0].imshow(diff_intensity, cmap='coolwarm', aspect='auto')
    axes[0].set_title(f'Intensity Difference\n({mod1} - {mod2})', fontsize=10)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # 梯度差图
    im2 = axes[1].imshow(diff_grad, cmap='coolwarm', aspect='auto')
    axes[1].set_title(f'Gradient Magnitude Difference\n({mod1} - {mod2})', fontsize=10)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"差异可视化图像已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == '__main__':
    # 示例数据路径（可根据实际情况修改）
    sample = {
         "t1":"/Users/wang/private/datasets/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData/BraTS-PED-00004-000/BraTS-PED-00004-000-t1n.nii.gz",
        "t1ce":"/Users/wang/private/datasets/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData/BraTS-PED-00004-000/BraTS-PED-00004-000-t1c.nii.gz",
        "t2":"/Users/wang/private/datasets/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData/BraTS-PED-00004-000/BraTS-PED-00004-000-t2w.nii.gz",
        "flair":"/Users/wang/private/datasets/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData/BraTS-PED-00004-000/BraTS-PED-00004-000-t2f.nii.gz",
        "seg":"/Users/wang/private/datasets/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData/BraTS-PED-00004-000/BraTS-PED-00004-000-seg.nii.gz"
    }

    # 数据预处理（保持与原代码一致）
    transforms = get_transform(sample.keys(), roi_size=(160, 176, 144))
    transforms = tfs.Compose(transforms)
    sample = load_and_transform(sample, transforms, flatten=False)

    # 绘制 t1 与 t2 的差异图
    plot_modality_differences(
        sample=sample,
        mod1='t1',
        mod2='t2',
        save_path='./output_contour/diff_visualization.png'
    )