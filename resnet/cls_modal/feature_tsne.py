import os
import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from monai.networks.nets import resnet50
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    Orientationd, EnsureTyped, SpatialPadd, 
    CenterSpatialCropd, ScaleIntensityRangePercentilesd
)
from monai.data import DataLoader, Dataset

# --- 配置 ---
CHECKPOINT_PATH = "./checkpoints/model_epoch_52.pth"
JSON_PATH = "./brats_modality_test.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. 加载模型并处理为特征提取模式 ---
def get_feature_extractor(checkpoint_path):
    # 初始化 ResNet-50
    model = resnet50(spatial_dims=3, n_input_channels=1, num_classes=4)
    # 加载权重
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    
    # 核心步骤：将全连接层替换为恒等映射，从而输出 GAP 后的 2048 维特征
    model.fc = nn.Identity() 
    model.to(DEVICE)
    model.eval()
    return model

# --- 2. 准备数据加载器 (使用验证集的变换) ---
val_transforms = Compose([
    LoadImaged(keys=["image"], image_only=True, ensure_channel_first=True),
    Orientationd(keys=['image'], axcodes="RAS"),
    EnsureTyped(keys=['image'], dtype=np.float32),
    SpatialPadd(keys=['image'], spatial_size=(160, 176, 144), mode='constant'),
    CenterSpatialCropd(keys=['image'], roi_size=(160, 176, 144)),
    ScaleIntensityRangePercentilesd(keys=['image'], lower=0.5, upper=99.5, b_min=-1, b_max=1, clip=True, relative=False, channel_wise=True),
])

with open(JSON_PATH, "r") as f:
    data_list = json.load(f)

# 这里建议使用 val 集合或者从 train 中抽样，以免内存溢出
val_ds = Dataset(data=data_list["val"], transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

# --- 3. 提取特征 ---
extractor = get_feature_extractor(CHECKPOINT_PATH)
features = []
labels = []

print("正在提取特征...")
with torch.no_grad():
    for data in tqdm(val_loader):
        img = data["image"].to(DEVICE)
        label = data["label"].item()
        
        # 得到 [1, 2048] 的特征向量
        feat = extractor(img)
        features.append(feat.cpu().numpy().flatten())
        labels.append(label)

features = np.array(features)
labels = np.array(labels)

# --- 4. 执行 t-SNE 降维 ---
print(f"正在执行 t-SNE (输入维度: {features.shape})...")
tsne = TSNE(
    n_components=2, 
    perplexity=30, 
    learning_rate='auto', 
    init='pca', 
    random_state=42
)
features_2d = tsne.fit_transform(features)

# --- 5. 可视化 ---
plt.figure(figsize=(10, 8))
classes = ['T1', 'T1CE', 'T2', 'FLAIR']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i in range(len(classes)):
    indices = np.where(labels == i)
    plt.scatter(
        features_2d[indices, 0], 
        features_2d[indices, 1], 
        c=colors[i], 
        label=classes[i], 
        alpha=0.6, 
        edgecolors='w', 
        s=50
    )


plt.legend()
plt.title("t-SNE Visualization of MRI Modalities (ResNet-50 Features)")
plt.xlabel("t-SNE dimension 1")
plt.ylabel("t-SNE dimension 2")
plt.grid(True, linestyle='--', alpha=0.5)
os.makedirs("./images", exist_ok=True)
plt.savefig("./images/modality_tsne.png", dpi=300)
plt.show()

print("可视化图像已保存为 modality_tsne.png")