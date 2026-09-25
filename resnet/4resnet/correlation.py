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
from monai.networks.nets import resnet18
from monai.transforms import (
    Compose, LoadImaged, Orientationd, EnsureTyped, 
    SpatialPadd, CenterSpatialCropd, ScaleIntensityRangePercentilesd
)
from monai.data import DataLoader, Dataset

# ==========================================
# 1. 配置参数
# ==========================================
CONFIG = {
    # 严格按照你训练的顺序排列：T1, T1CE, T2, FLAIR
    "model_paths": [
        "experiments/4resnet_t1/best_model.pth",
        "experiments/4resnet_t1ce/best_model.pth",
        "experiments/4resnet_t2/best_model.pth",
        "experiments/4resnet_flair/best_model.pth"
    ],
    # 注意：这里必须使用包含 4 个模态路径列表的联合 JSON，以保证提取的是同一个病人的 4 个模态
    "json_path": "brats2018.json", 
    "save_dir": "experiments/analysis_4resnet",
    "num_samples": 50,
    "roi_size": (160, 176, 144),
    "modality_names": ['T1', 'T1CE', 'T2', 'FLAIR'],
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)

# ==========================================
# 2. 特征提取器定义
# ==========================================
def load_feature_extractor(checkpoint_path, device):
    """加载独立的单通道 ResNet18 模型，并返回特征提取器"""
    model = resnet18(spatial_dims=3, n_input_channels=1, num_classes=2)
    
    # 兼容处理：检查是否存在文件
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到权重文件: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint if 'model_state_dict' not in checkpoint else checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    
    # 替换最后的全连接层，直接输出 GAP 后的 512 维特征
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model

# ==========================================
# 3. 核心分析逻辑
# ==========================================
def run_4resnet_rsa():
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

    dataset = Dataset(data=data_list["train"], transform=val_transforms)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # --- B. 实例化 4 个独立的模型 ---
    print(f"正在加载 4 个独立的 ResNet18 模型...")
    extractors = [
        load_feature_extractor(path, CONFIG["device"]) 
        for path in CONFIG["model_paths"]
    ]

    all_corr_matrices = []

    # --- C. 特征提取与比对 ---
    print(f"开始提取特征并计算相关性 (目标样本量: {CONFIG['num_samples']})...")
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(loader)):
            if i >= CONFIG["num_samples"]:
                break
            
            # images shape: (1, 4, D, H, W)
            inputs = batch_data["images"].to(CONFIG["device"])
            sample_features = []
            
            for m_idx in range(4):
                # 1. 切片出当前模态，保持形状为 (1, 1, D, H, W)
                single_modality = inputs[:, m_idx:m_idx+1, ...]
                
                # 2. 喂入该模态专属的 ResNet18 模型
                feat = extractors[m_idx](single_modality)
                
                # 3. L2 特征归一化 (RSA 必备)
                feat_np = feat.cpu().numpy().flatten()
                feat_norm = feat_np / (np.linalg.norm(feat_np) + 1e-8)
                sample_features.append(feat_norm)
            
            # 计算 4 个特化特征之间的 Pearson 相关系数
            corr_matrix = np.corrcoef(np.array(sample_features))
            all_corr_matrices.append(corr_matrix)

    # 计算均值矩阵
    mean_corr_matrix = np.mean(all_corr_matrices, axis=0)

    # ==========================================
    # 4. 调用统一绘图函数
    # ==========================================
    save_path = os.path.join(CONFIG["save_dir"], "rsa_matrix_4resnets.png")
    
    plot_rsa_matrix(
        matrix=mean_corr_matrix,
        labels=CONFIG["modality_names"],
        save_path=save_path,
        title="Representational Similarity Analysis (RSA)\n4 Distinct ResNet-18 Models",
        cmap="viridis"
    )

    print(f"\n[完成] 4-ResNets 联合相关性矩阵已保存至: {save_path}")

if __name__ == "__main__":
    run_4resnet_rsa()