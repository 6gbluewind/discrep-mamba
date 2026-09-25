import matplotlib.pyplot as plt
import numpy as np
import pywt
import os
import nibabel as nib
from utils import load_nifit, get_transform, load_and_transform
from monai import transforms as tf

def visualize_coeffs(coeffs_list, save_dir='./wavelet_coeffs_visualization_trans_t1c'):
    """visualze wavelet decomposition coefficient"""
    
    # create save dir
    os.makedirs(save_dir, exist_ok=True)
    
    # 1.Visualized wavelet decomposition coefficients (coeffs_list[0])
    a = coeffs_list[0]
    print(f"Approximate coefficient shape: {a.shape}")
    
    # get slices
    mid_z = a.shape[2] // 2
    mid_y = a.shape[1] // 2
    mid_x = a.shape[0] // 2
    
    fig_a, axes_a = plt.subplots(1, 3, figsize=(15, 5))
    fig_a.suptitle(f'Level {len(coeffs_list)-1} Approximation Coefficients (Shape: {a.shape})', fontsize=16)
    
    # XY plane
    im_xy = axes_a[0].imshow(a[:, :, mid_z], cmap='gray', aspect='auto')
    axes_a[0].set_title(f'XY Plane (z={mid_z})')
    axes_a[0].set_xlabel('X')
    axes_a[0].set_ylabel('Y')
    plt.colorbar(im_xy, ax=axes_a[0], fraction=0.046, pad=0.04)
    
    # XZ plane
    im_xz = axes_a[1].imshow(a[:, mid_y, :], cmap='gray', aspect='auto')
    axes_a[1].set_title(f'XZ Plane (y={mid_y})')
    axes_a[1].set_xlabel('Z')
    axes_a[1].set_ylabel('X')
    plt.colorbar(im_xz, ax=axes_a[1], fraction=0.046, pad=0.04)
    
    # YZ plane
    im_yz = axes_a[2].imshow(a[mid_x, :, :], cmap='gray', aspect='auto')
    axes_a[2].set_title(f'YZ Plane (x={mid_x})')
    axes_a[2].set_xlabel('Z')
    axes_a[2].set_ylabel('Y')
    plt.colorbar(im_yz, ax=axes_a[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    a_save_path = os.path.join(save_dir, f'approximation_coeffs_level_{len(coeffs_list)-1}.png')
    plt.savefig(a_save_path, dpi=150, bbox_inches='tight')
    print(f"Approximate coefficient save to : {a_save_path}")
    plt.show()
    
    # 2. Visual detail coefficient
    print("\n=== Visual detail coefficient ===")
    d = coeffs_list[1]
    
    # get all keys
    directions = list(d.keys())
    print(f"detail coefficient all keys: {directions}")
    
    for direction in directions:
        coeff = d[direction]
        print(f"{direction} shape: {coeff.shape}")
        
        fig_d, axes_d = plt.subplots(1, 3, figsize=(15, 5))
        fig_d.suptitle(f'Level {len(coeffs_list)-1} Detail Coefficients - Direction: {direction} (Shape: {coeff.shape})', fontsize=16)
        
        # XY plane
        im_xy = axes_d[0].imshow(coeff[:, :, mid_z], cmap='seismic', aspect='auto', 
                                 vmin=-np.max(np.abs(coeff)), vmax=np.max(np.abs(coeff)))
        axes_d[0].set_title(f'XY Plane (z={mid_z})')
        axes_d[0].set_xlabel('X')
        axes_d[0].set_ylabel('Y')
        plt.colorbar(im_xy, ax=axes_d[0], fraction=0.046, pad=0.04)
        
        # XZ plane
        im_xz = axes_d[1].imshow(coeff[:, mid_y, :], cmap='seismic', aspect='auto',
                                 vmin=-np.max(np.abs(coeff)), vmax=np.max(np.abs(coeff)))
        axes_d[1].set_title(f'XZ Plane (y={mid_y})')
        axes_d[1].set_xlabel('Z')
        axes_d[1].set_ylabel('X')
        plt.colorbar(im_xz, ax=axes_d[1], fraction=0.046, pad=0.04)
        
        # YZ plane
        im_yz = axes_d[2].imshow(coeff[mid_x, :, :], cmap='seismic', aspect='auto',
                                 vmin=-np.max(np.abs(coeff)), vmax=np.max(np.abs(coeff)))
        axes_d[2].set_title(f'YZ Plane (x={mid_x})')
        axes_d[2].set_xlabel('Z')
        axes_d[2].set_ylabel('Y')
        plt.colorbar(im_yz, ax=axes_d[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        d_save_path = os.path.join(save_dir, f'detail_coeffs_{direction}_level_{len(coeffs_list)-1}.png')
        plt.savefig(d_save_path, dpi=150, bbox_inches='tight')
        print(f"detail coefficient {direction} save to: {d_save_path}")
        plt.show()
    
    # 3. detail coefficient and approximate coefficient
    print("\n=== create all figure ===")
    fig_all, axes_all = plt.subplots(2, 4, figsize=(20, 10))
    fig_all.suptitle(f'All Detail Coefficients - Level {len(coeffs_list)-1} (XY Plane at z={mid_z})', fontsize=20)
    
    axes_all = axes_all.flatten()
    
    for idx, direction in enumerate(directions):
        coeff = d[direction]
        
        # only show XY plane slice
        im = axes_all[idx].imshow(coeff[:, :, mid_z], cmap='seismic', aspect='auto',
                                  vmin=-np.max(np.abs(coeff)), vmax=np.max(np.abs(coeff)))
        axes_all[idx].set_title(f'{direction}')
        axes_all[idx].set_xlabel('X')
        axes_all[idx].set_ylabel('Y')
        plt.colorbar(im, ax=axes_all[idx], fraction=0.046, pad=0.04)
    
    # last subplt show  approximate coefficient
    axes_all[-1].imshow(a[:, :, mid_z], cmap='gray', aspect='auto')
    axes_all[-1].set_title('Approximation (a)')
    axes_all[-1].set_xlabel('X')
    axes_all[-1].set_ylabel('Y')
    plt.colorbar(im_xy, ax=axes_all[-1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    all_save_path = os.path.join(save_dir, f'all_coeffs_summary_level_{len(coeffs_list)-1}.png')
    plt.savefig(all_save_path, dpi=150, bbox_inches='tight')
    print(f"all figure save to: {all_save_path}")
    plt.show()
    
    return a_save_path, save_dir


def process_and_visualize_volume(volume, levels=3):
    
    print("volume shape:", volume.shape)
    
    coeffs_list = pywt.wavedecn(volume, 'db2', level=levels)
    
    a = coeffs_list[0]  # approximate coefficient
    d = coeffs_list[1]  # detail coefficient
    
    # visual and save
    save_paths = visualize_coeffs(coeffs_list)
    
    return coeffs_list, a, d

if __name__ == "__main__":

    # image = nib.load("/media/bmp1/store5/dataset/brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1c.nii.gz")
    # fdata = image.get_fdata()
    # fdata = load_nifit("/media/bmp1/store5/dataset/brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1n.nii.gz",
    #                    to_array=True)
    
    sample = {
        't1ce': "/media/bmp1/store5/dataset/brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1c.nii.gz"
    }
    trans = get_transform(['t1ce'], roi_size=(240, 240, 155))
    trans = tf.Compose(trans)
    data = load_and_transform(sample, trans)
    fdata =data['t1ce'][0]
   
    coeffs_list, a, d = process_and_visualize_volume(fdata, levels=3)
    