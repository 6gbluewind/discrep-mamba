import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

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
    "save_dir": "results/analysis_pca",
    "num_samples": 80,         # 样本量
    "n_components": 50,       # 提取的主成分数量
    "roi_size": (160, 176, 144),
    "modality_names": ['T1', 'T1CE', 'T2', 'FLAIR'],
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)

# ==========================================
# 2. PCA RSA 核心逻辑
# ==========================================
def run_pca_rsa_analysis():
    # --- A. 准备数据变换 (保持与深度学习实验一致) ---
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

    dataset = Dataset(data=data_list["val"], transform=val_transforms)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # --- B. 数据收集与展平 ---
    modality_data = [[] for _ in range(4)]

    print(f"正在读取数据并展平体素 (目标样本量: {CONFIG['num_samples']})...")
    for i, batch_data in enumerate(tqdm(loader)):
        if i >= CONFIG["num_samples"]:
            break
        
        # images shape: (1, 4, D, H, W)
        img_tensor = batch_data["images"][0].numpy()
        
        for m_idx in range(4):
            # 将 3D 图像拉平为 1D 向量
            flat_img = img_tensor[m_idx].flatten()
            modality_data[m_idx].append(flat_img)

    # --- C. 执行 PCA 降维 ---
    modality_features = []
    
    print("\n正在对每个模态执行 PCA 降维...")
    for m_idx in range(4):
        matrix = np.array(modality_data[m_idx])
        n_samples_actual = matrix.shape[0]
        
        # 核心修正：确保 n_components 不超过样本数限制
        current_n = min(n_samples_actual - 1, CONFIG["n_components"])
        
        pca = PCA(n_components=current_n, random_state=42)
        reduced_data = pca.fit_transform(matrix)
        
        # 对降维后的特征进行 L2 归一化，增强相关性计算稳定性 (RSA 标准操作)
        norms = np.linalg.norm(reduced_data, axis=1, keepdims=True)
        reduced_data = reduced_data / (norms + 1e-8)
        
        modality_features.append(reduced_data)
        print(f"  - {CONFIG['modality_names'][m_idx]} 完成。累计解释方差: {np.sum(pca.explained_variance_ratio_):.4f}")

    # --- D. 计算模态间相关性矩阵 ---
    final_corr_matrix = np.zeros((4, 4))
    
    print("\n正在计算 PCA 特征相关性矩阵...")
    for i in range(4):
        for j in range(4):
            # 计算两个模态 PCA 特征分布的 Pearson 相关系数
            r = np.corrcoef(modality_features[i].flatten(), modality_features[j].flatten())[0, 1]
            final_corr_matrix[i, j] = r

    # ==========================================
    # 4. 调用统一绘图函数
    # ==========================================
    save_path = os.path.join(CONFIG["save_dir"], "pca_rsa_matrix.png")
    
    plot_rsa_matrix(
        matrix=final_corr_matrix,
        labels=CONFIG["modality_names"],
        save_path=save_path,
        title=f"RSA Matrix: PCA Statistical Baseline\n(Top {CONFIG['n_components']} Components)",
        cmap="coolwarm"  # Baseline 通常建议用 coolwarm 区分于深度学习结果
    )
    
    print(f"\n[完成] PCA 基准相关性矩阵已保存至: {save_path}")

if __name__ == "__main__":
    run_pca_rsa_analysis()