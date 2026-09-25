import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# --- 路径兼容处理：确保能找到父目录下的 draw.py ---
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from draw import plot_rsa_matrix

# MONAI 相关库
from monai.networks.nets import resnet50
from monai.transforms import (
    Compose, LoadImaged, Orientationd, EnsureTyped, 
    SpatialPadd, CenterSpatialCropd, ScaleIntensityRangePercentilesd
)
from monai.data import DataLoader, Dataset

# ==========================================
# 1. 配置参数
# ==========================================
CONFIG = {
    "checkpoint_path": "Independent/checkpoints/model_epoch_52.pth",
    "json_path": "brats2018.json",
    "save_dir": "results/analysis",
    "num_samples": 100,
    "roi_size": (160, 176, 144),
    "modality_names": ['T1', 'T1CE', 'T2', 'FLAIR'],
    "device": torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)

# ==========================================
# 2. 特征提取器定义
# ==========================================
def get_feature_extractor(checkpoint_path, device):
    """加载预训练 ResNet-50 并提取 GAP 层特征"""
    model = resnet50(
        spatial_dims=3, 
        n_input_channels=1, # 对应单个模态输入
        num_classes=4
    )
    # 加载权重
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # 将全连接层替换为 Identity，直接输出 2048 维向量
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model

# ==========================================
# 3. 相关性矩阵计算
# ==========================================
def run_rsa_analysis():
    # --- A. 准备数据 ---
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

    # --- B. 加载模型 ---
    extractor = get_feature_extractor(CONFIG["checkpoint_path"], CONFIG["device"])
    
    all_corr_matrices = []

    # --- C. 特征提取循环 ---
    print(f"开始提取 ResNet-50 特征 (Device: {CONFIG['device']})...")
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(loader)):
            if i >= CONFIG["num_samples"]:
                break
            
            # inputs shape: (1, 4, D, H, W)
            inputs = batch_data["images"].to(CONFIG["device"])
            
            sample_features = []
            # 依次将 4 个通道（模态）喂入 1-channel 的 ResNet-50
            for m_idx in range(4):
                single_modality = inputs[:, m_idx:m_idx+1, ...]
                
                feat = extractor(single_modality)
                # L2 归一化（使相关性计算更稳定，类似于余弦相似度）
                feat_np = feat.cpu().numpy().flatten()
                feat_norm = feat_np / (np.linalg.norm(feat_np) + 1e-8)
                sample_features.append(feat_norm)
            
            # 计算 4 个特征向量两两之间的 Pearson 相关系数
            corr_matrix = np.corrcoef(np.array(sample_features))
            all_corr_matrices.append(corr_matrix)

    # 计算均值矩阵
    mean_corr_matrix = np.mean(all_corr_matrices, axis=0)

    # ==========================================
    # 4. 调用封装好的统一绘图函数
    # ==========================================
    save_path = os.path.join(CONFIG["save_dir"], "rsa_feature_matrix_resnet50.png")
    
    plot_rsa_matrix(
        matrix=mean_corr_matrix,
        labels=CONFIG["modality_names"],
        save_path=save_path,
        title="Representational Similarity Analysis (RSA)\nResNet-50 Latent Space",
        cmap="RdBu_r"
    )

    print(f"\n[完成] 独立训练 ResNet-50 RSA 矩阵已保存至: {save_path}")

if __name__ == "__main__":
    run_rsa_analysis()