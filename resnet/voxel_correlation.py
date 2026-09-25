import os
import sys
import json
import numpy as np
from tqdm import tqdm

# --- 路径兼容处理：确保能找到父目录下的 draw.py ---
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from draw import plot_rsa_matrix

# MONAI 相关库
from monai.transforms import (
    Compose, LoadImaged, Orientationd, EnsureTyped, 
    SpatialPadd, CenterSpatialCropd, ScaleIntensityRangePercentilesd
)
from monai.data import DataLoader, Dataset

# ==========================================
# 1. 配置参数
# ==========================================
CONFIG = {
    "json_path": "brats2018.json",
    "save_dir": "experiments/analysis_voxel/result",
    "num_samples": 50,                   # 分析样本数
    "roi_size": (160, 176, 144),
    "modality_names": ['T1', 'T1CE', 'T2', 'FLAIR'],
    "bg_threshold": -0.95  # 关键参数：背景过滤阈值
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)

# ==========================================
# 2. 核心分析逻辑
# ==========================================
def run_voxel_rsa():
    # --- A. 准备数据 ---
    # 严格保持与深度学习实验完全一致的预处理，确保变量单一
    val_transforms = Compose([
        LoadImaged(keys=["images"], image_only=True, ensure_channel_first=True),
        Orientationd(keys=['images'], axcodes="RAS"),
        EnsureTyped(keys=['images'], dtype=np.float32),
        SpatialPadd(keys=['images'], spatial_size=CONFIG["roi_size"], mode='constant'),
        CenterSpatialCropd(keys=['images'], roi_size=CONFIG["roi_size"]),
        ScaleIntensityRangePercentilesd(
            keys=['images'], lower=0.5, upper=99.5, 
            b_min=-1, b_max=1, clip=True, channel_wise=True
        ),
    ])

    with open(CONFIG["json_path"], "r") as f:
        data_list = json.load(f)

    dataset = Dataset(data=data_list["train"], transform=val_transforms)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_corr_matrices = []

    print(f"开始提取体素特征并计算相关性 (目标样本量: {CONFIG['num_samples']})...")
    
    for i, batch_data in enumerate(tqdm(loader)):
        if i >= CONFIG["num_samples"]:
            break
        
        # 提取当前样本，去掉 Batch 维度。shape: (4, D, H, W)
        img_volume = batch_data["images"][0].numpy()
        
        # --- B. 背景过滤 (Background Masking) ---
        # 【极度重要】由于 MRI 图像周围有大量黑色背景，且不同模态的背景像素值都是最低值(-1左右)。
        # 如果把背景也算进相关性里，所有模态的相关性都会被拉高到 0.9 以上。
        # 这里我们利用 T1CE (索引为1) 或平均值来生成一个大脑前景掩码 (Foreground Mask)
        mean_volume = np.mean(img_volume, axis=0)
        brain_mask = mean_volume > CONFIG["bg_threshold"]
        
        sample_voxels = []
        for m_idx in range(4):
            # 利用脑部掩码提取前景体素，并自动拉平成 1D 向量
            fg_voxels = img_volume[m_idx][brain_mask]
            sample_voxels.append(fg_voxels)
        
        # 将当前病人的 4 个一维体素向量堆叠，计算 Pearson 相关系数矩阵
        corr_matrix = np.corrcoef(np.array(sample_voxels))
        all_corr_matrices.append(corr_matrix)

    # 计算均值矩阵
    mean_corr_matrix = np.mean(all_corr_matrices, axis=0)

    # ==========================================
    # 3. 调用统一绘图函数
    # ==========================================
    save_path = os.path.join(CONFIG["save_dir"], "rsa_matrix_raw_voxels.png")
    
    plot_rsa_matrix(
        matrix=mean_corr_matrix,
        labels=CONFIG["modality_names"],
        save_path=save_path,
        title="Representational Similarity Analysis (RSA)\nRaw Voxel Space (Foreground Only)",
        cmap="RdBu_r"  # 基准测试建议使用 coolwarm
    )

    print(f"\n[完成] 原始体素级 RSA 矩阵已保存至: {save_path}")

if __name__ == "__main__":
    run_voxel_rsa()