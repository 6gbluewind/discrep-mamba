import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from monai import transforms as tfs
from skimage import measure
from utils import data_analysis, standardization, normaliation
from utils import load_nifit, create_logger, load_and_transform, get_transform
from typing import Dict, Optional, Union, List
from scipy.ndimage import gaussian_filter, sobel

matplotlib.use('Agg')

log_dir = './output_contour'
os.makedirs(log_dir, exist_ok=True)

# create logger
logger = create_logger(logging_dir=log_dir)

def visualization(
    sample: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    prefix: Optional[str] = None,
    layout: str = 'horizontal',  # 'horizontal', 'vertical', 'grid'
    show_histogram: bool = False,
    norm: str = None, # 'minmax', 'stand', None
):  
    modalities = list(sample.keys())
    n_modalities = len(modalities)
    
    # 确定子图布局
    if show_histogram:
        n_rows = 2
    else:
        n_rows = 1
    
    if layout == 'vertical':
        n_cols = 1
        figsize = (8, 4 * n_modalities * n_rows)
    elif layout == 'grid':
        n_cols = int(np.ceil(np.sqrt(n_modalities)))
        n_rows_actual = int(np.ceil(n_modalities / n_cols))
        if show_histogram:
            n_rows_actual *= 2
        figsize = (5 * n_cols, 4 * n_rows_actual)
    else:  # horizontal
        n_cols = n_modalities
        figsize = (4 * n_modalities, 4 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    slice_idx = sample[modalities[0]].shape[2] // 2
    
    for i, modality in enumerate(modalities):
        data = sample[modality]

        if norm == 'minmax':
            # logger.info('use minmax normalization')
            data = normaliation(data)
            # print(data.max(), data.min())
        elif norm == 'stand':
            # logger.info('use standardization normalization')
            data = standardization(data)
            # print(data.max(), data.min())
        
        slice_img = data[:, :, slice_idx]


        # 显示图像
        row_idx = 0
        col_idx = i if layout == 'horizontal' else i % n_cols
        if layout == 'grid':
            row_idx = (i // n_cols) * 2
            col_idx = i % n_cols
        
        ax_img = axes[row_idx, col_idx]
        im = ax_img.imshow(slice_img, cmap='gray', aspect='auto')
        
        ax_img.set_title(f'{modality}\nSlice Z={slice_idx}', fontsize=10)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        
        # add colorbar
        plt.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
        
        # show histogram
        if show_histogram:
            hist_row = row_idx + 1 if layout in ['horizontal', 'vertical'] else row_idx + 1
            ax_hist = axes[hist_row, col_idx]
            
            # reomve background pixel
            data = slice_img.flatten()
            data_filtered = data[data > np.percentile(data, 1)]

            ax_hist.hist(data_filtered, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
            ax_hist.set_title(f'{modality} Intensity Distribution', fontsize=9)
            ax_hist.set_xlabel('Intensity')
            ax_hist.set_ylabel('Frequency')
            ax_hist.grid(True, alpha=0.3)
            
            mean_val = np.mean(slice_img)
            median_val = np.median(slice_img)
            ax_hist.axvline(mean_val, color='red', linestyle='--', linewidth=1, label=f'Mean: {mean_val:.1f}')
            ax_hist.axvline(median_val, color='green', linestyle='--', linewidth=1, label=f'Median: {median_val:.1f}')
            ax_hist.legend(fontsize=8)
    
    # 隐藏多余的子图
    if layout == 'grid':
        total_cells = n_rows * n_cols
        used_cells = n_modalities * (2 if show_histogram else 1)
        for idx in range(used_cells, total_cells):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # save 
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = prefix or "visualization"
        filename += f"_{layout}"
        if show_histogram:
            filename += "_with_hist"
        filename += ".png"
        
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        print(f"可视化图像已保存到: {filepath}")
    else:
        plt.show()
    
    plt.close()

def contour_analysis(img, save_path=None):    
    h, w = img.shape
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 原始图像 + 等值线
    ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
    im1 = ax1.imshow(img, cmap='gray', aspect='auto')
    contours = measure.find_contours(img, level=0.1)
    for contour in contours: 
        ax1.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=1, alpha=0.7)
    ax1.set_title('Original Image with Detected Contours')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # 2. 梯度幅度
    ax2 = plt.subplot2grid((3, 4), (0, 2))
    grad_x = sobel(img, axis=1)
    grad_y = sobel(img, axis=0)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    im2 = ax2.imshow(grad_magnitude, cmap='hot', aspect='auto')
    ax2.set_title('Gradient Magnitude')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # 3. 等值线图 100 level
    ax3 = plt.subplot2grid((3, 4), (0, 3))
    min_val, max_val = img.min(), img.max()
    levels = np.linspace(min_val, max_val, 100)
    contourf = ax3.contourf(X, Y, img, levels=levels, cmap='Spectral') # with fill
    ax3.set_title('Filled Contour')
    plt.colorbar(contourf, ax=ax3, shrink=0.8)
    
    # 4. 3D等值线图
    ax4 = plt.subplot2grid((3, 4), (1, 2), projection='3d')
    contour_lines_3d = ax4.contour3D(X, Y, img, 20, cmap='viridis')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Intensity')
    ax4.set_title('3D Contour')
    ax4.view_init(elev=30, azim=45)
    
    # 5. 等值线统计
    ax5 = plt.subplot2grid((3, 4), (1, 3))
    # 计算不同水平的等值线长度
    contour_lengths = []
    contour_areas = []
    level_range = np.linspace(min_val, max_val, 10)
    
    for level in level_range:
        contours = measure.find_contours(img, level=level)
        total_length = 0
        total_area = 0
        
        for contour in contours:
            if len(contour) > 2:
                # 近似计算长度
                lengths = np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1))
                total_length += np.sum(lengths)
                
                # 近似计算面积（使用多边形面积公式）
                x_vals = contour[:, 1]
                y_vals = contour[:, 0]
                area = 0.5 * np.abs(np.dot(x_vals, np.roll(y_vals, 1)) - 
                                   np.dot(y_vals, np.roll(x_vals, 1)))
                total_area += area
        
        contour_lengths.append(total_length)
        contour_areas.append(total_area)
    
    ax5.plot(level_range, contour_lengths, 'b-o', label='Total Contour Length')
    ax5.set_xlabel('Contour Level')
    ax5.set_ylabel('Length (pixels)', color='b')
    ax5.tick_params(axis='y', labelcolor='b')
    ax5.grid(True, alpha=0.3)
    
    ax5_twin = ax5.twinx()
    ax5_twin.plot(level_range, contour_areas, 'r-s', label='Total Area')
    ax5_twin.set_ylabel('Area (pixels²)', color='r')
    ax5_twin.tick_params(axis='y', labelcolor='r')
    
    ax5.set_title('Contour Statistics vs Level')
    
    # 6. 特定等高线分析
    ax6 = plt.subplot2grid((3, 4), (2, 0), colspan=2)
    # 选择几个关键水平
    key_levels = [min_val + (max_val - min_val) * i/4 for i in range(5)]
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    
    for level, color in zip(key_levels, colors):
        contours = measure.find_contours(img, level=level)
        for contour in contours[:3]:  # 每个水平最多显示3个轮廓
            ax6.plot(contour[:, 1], contour[:, 0], color=color, 
                    linewidth=2, alpha=0.7, label=f'Level {level:.1f}')
    
    ax6.set_xlim(0, w)
    ax6.set_ylim(h, 0)  # 反转Y轴以匹配图像坐标
    ax6.set_title('Key Contour Levels')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.legend(loc='upper right', fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # 7. 等值线属性表格
    ax7 = plt.subplot2grid((3, 4), (2, 2), colspan=2)
    ax7.axis('tight')
    ax7.axis('off')
    
    # 计算统计信息
    stats_data = [
        ['Image Shape', f'{h} × {w}'],
        ['Min Intensity', f'{img.min():.2f}'],
        ['Max Intensity', f'{img.max():.2f}'],
        ['Mean Intensity', f'{img.mean():.2f}'],
        ['Std Intensity', f'{img.std():.2f}'],
        ['Total Contours', f'{len(contours)}'],
        ['Mean Gradient', f'{grad_magnitude.mean():.3f}'],
    ]
    
    table = ax7.table(cellText=stats_data, 
                      colLabels=['Property', 'Value'],
                      cellLoc='left',
                      loc='center',
                      colWidths=[0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.suptitle('Comprehensive Contour Analysis', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Advanced contour analysis saved to: {save_path}")
    
    # plt.show()
    return fig

if __name__ == '__main__':
    # sample = {
    #     't1': '/media/bmp1/store5/dataset/brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1n.nii.gz',
    #     't1ce': '/media/bmp1/store5/dataset/brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1c.nii.gz',
    #     't2': '/media/bmp1/store5/dataset/brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2w.nii.gz',
    #     'flair': '/media/bmp1/store5/dataset/brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2f.nii.gz'
    # }
    # /Users/wang/private/datasets/BraTS2023-GLI/BraTS-GLI-00000-000

    sample = {
        't1': '/Users/wang/private/datasets/BraTS2023-GLI/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1n.nii.gz',
        't1ce': '/Users/wang/private/datasets/BraTS2023-GLI/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1c.nii.gz',
        't2': '/Users/wang/private/datasets/BraTS2023-GLI/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2w.nii.gz',
        'flair': '/Users/wang/private/datasets/BraTS2023-GLI/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2f.nii.gz'
    }

    transforms = get_transform(sample.keys(), roi_size=(160, 176, 144))
    transforms = tfs.Compose(transforms)
    sample = load_and_transform(sample, transforms, flatten=False)

    # surface_graph(sample['t1'][:,:,77])
    for modality_name, modality_val in sample.items():
        contour_analysis(modality_val[0,:,:,77], save_path=f'./output_contour/contour_{modality_name}.png')
