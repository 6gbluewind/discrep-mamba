import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai import transforms as tfs
from skimage import measure
from utils import data_analysis, squeeze
from utils import load_nifit, create_logger, load_and_transform, get_transform, get_samples
from typing import Dict, Optional, Union, List
from scipy.ndimage import gaussian_filter, sobel

matplotlib.use('Agg')

log_dir = './output_surface'
os.makedirs(log_dir, exist_ok=True)

# create logger
logger = create_logger(logging_dir=log_dir)


def surface_graph(data, save_path='./output_surface'):
    os.makedirs(save_path, exist_ok=True)
    
    if isinstance(data, np.ndarray):
        h, w, d = data.shape
        img = data[:, :, d//2]
        
        
        x = np.arange(w)  
        y = np.arange(h)  
        X, Y = np.meshgrid(x, y)
        
        fig = plt.figure(figsize=(12, 8))
        # 3D surface graph
        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(X, Y, img, 
                                cmap='viridis',
                                edgecolor='none',
                                alpha=0.8,
                                rstride=2, 
                                cstride=2)
        ax1.set_xlabel('X Position (pixels)')
        ax1.set_ylabel('Y Position (pixels)')
        ax1.set_zlabel('Pixel Intensity')
        ax1.set_title('3D Surface Plot: Intensity vs Position')
        fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5, label='Intensity')
        
        # 添加2D图像作为对比
        ax2 = fig.add_subplot(122)
        im2 = ax2.imshow(img, cmap='viridis', aspect='auto')
        ax2.set_xlabel('X Position (pixels)')
        ax2.set_ylabel('Y Position (pixels)')
        ax2.set_title('2D Slice View')
        fig.colorbar(im2, ax=ax2, shrink=0.5, aspect=5, label='Intensity')
        
        plt.tight_layout()
        filename = '3d_surface_graph_single.png' 
    elif isinstance(data, dict):
        modalities = list(data.keys())
        n_modalities = len(modalities)
        
        n_rows = (n_modalities + 1) // 2 
        n_cols = min(2, n_modalities)
        
        fig_height = 6 * n_rows
        fig = plt.figure(figsize=(15, fig_height))
        
        for idx, (modality_name, modality_data) in enumerate(data.items()):
            if not isinstance(modality_data, np.ndarray):
                modality_data = np.array(modality_data)
                
            assert len(modality_data.shape) == 3, 'modality data must 3 dimensional'
                
            # middle slice
            h, w, d = modality_data.shape
            img = modality_data[:, :, d//2]
            
            # create grid 
            x = np.arange(w)  
            y = np.arange(h)  
            X, Y = np.meshgrid(x, y)
            
            # create 3D subplot
            ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
            
            # drew 3D surface
            surf = ax.plot_surface(
                X, Y, img,
                cmap='viridis',
                edgecolor='none',
                alpha=0.8,
                rstride=2,
                cstride=2
            )
            # set label and title
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.set_zlabel('Intensity')
            ax.set_title(f'{modality_name}: {h}×{w}×{d}')
            # add color bar
            cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1)
            cbar.set_label('Intensity')
            # view point
            ax.view_init(elev=30, azim=45)
        
        plt.suptitle('Multi-Modality 3D Surface Plots', fontsize=16, y=0.98)
        plt.tight_layout()
        filename = '3d_surface_graph_multi_modal.png'
    
    else:
        raise TypeError("Input data must be either numpy.ndarray or dict")
    
    # save image
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    # print(f"3D surface graph saved to: {filepath}")
    return filepath


def main(data_root):
    transforms = get_transform(['t1', 't1ce', 't2', 'flair'], roi_size=(160, 176, 144))
    transforms = tfs.Compose(transforms)

    samples, sample_names = get_samples(data_root)
    idx = 0
    for sample in tqdm(samples):
        sample = load_and_transform(sample, transforms, flatten=False)
        sample = squeeze(sample)
        surface_graph(sample, save_path = f'./output_surface/{sample_names[idx]}')
        idx += 1


if __name__ == '__main__':
    # sample = {
    #     't1': '/media/bmp1/store5/dataset/brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1n.nii.gz',
    #     't1ce': '/media/bmp1/store5/dataset/brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1c.nii.gz',
    #     't2': '/media/bmp1/store5/dataset/brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2w.nii.gz',
    #     'flair': '/media/bmp1/store5/dataset/brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2f.nii.gz'
    # }
    # /Users/wang/private/datasets/BraTS2023-GLI/BraTS-GLI-00000-000

    # 单个样本的 surface 图
    # sample = {
    #     't1': '/Users/wang/private/datasets/BraTS2023-GLI/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1n.nii.gz',
    #     't1ce': '/Users/wang/private/datasets/BraTS2023-GLI/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1c.nii.gz',
    #     't2': '/Users/wang/private/datasets/BraTS2023-GLI/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2w.nii.gz',
    #     'flair': '/Users/wang/private/datasets/BraTS2023-GLI/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2f.nii.gz'
    # }

    # transforms = get_transform(sample.keys(), roi_size=(160, 176, 144))
    # transforms = tfs.Compose(transforms)
    # sample = load_and_transform(sample, transforms, flatten=False)
    # sample = squeeze(sample)
    
    # surface_graph(sample, save_path = './output_surface',)

    # 所有样本的 surface 图
    data_root = '/media/bmp1/store5/dataset/brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
    main(data_root)