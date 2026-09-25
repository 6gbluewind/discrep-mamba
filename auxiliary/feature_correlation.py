import os
import sys
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# 确保能找到你的自定义模块
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils import load_checkpoint, create_loader
from models.multimae3d import instantiate_MultiMAE

@torch.no_grad()
def evaluate_and_plot_correlation(model, dataloader, device, save_dir="./logs/correlation"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    modalities = ['t1n', 't1c', 't2w', 't2f']
    # 用于存储每个样本的 4x4 相关性矩阵
    sample_corr_matrices = []

    pbar = tqdm(dataloader, desc="Extracting features for Correlation Analysis")
    
    for batch in pbar:
        if isinstance(batch, dict):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # 1. 提取特征
        output = model(batch, mask_ratio=0.0, return_image=False)
        last_layer_tokens = output["encoder_tokens"][-1] # [B, Total_Tokens, Dim]
        task_ranges = output["task_ranges"]

        # 获取当前 batch 的特征字典
        batch_features = {} # {modality: tensor of shape [B, Dim]}
        
        for task in modalities:
            if task in task_ranges:
                start, end = task_ranges[task]
                task_tokens = last_layer_tokens[:, start:end, :]
                # Mean Pooling 得到模态全局表征
                pooled_features = task_tokens.mean(dim=1) 
                # L2 归一化使相关性计算更稳定（等同于余弦相似度）
                pooled_features = torch.nn.functional.normalize(pooled_features, p=2, dim=1)
                batch_features[task] = pooled_features.cpu().numpy()

        # 2. 对 Batch 内的每个样本计算 4x4 矩阵
        batch_size = batch_features[modalities[0]].shape[0]
        for b in range(batch_size):
            # 提取该样本的 4 个模态特征，组合成 [4, Dim]
            modality_vectors = np.stack([batch_features[m][b] for m in modalities])
            
            # 计算 Pearson 相关系数矩阵 (4x4)
            # np.corrcoef 会返回 4x4 矩阵
            corr_matrix = np.corrcoef(modality_vectors)
            sample_corr_matrices.append(corr_matrix)

    # 3. 计算所有样本的平均相关性矩阵
    mean_corr_matrix = np.mean(sample_corr_matrices, axis=0)

    # 4. 绘制热力图
    plt.figure(figsize=(9, 7))
    sns.set_theme(style="white")
    
    # 使用标注显示数值
    ax = sns.heatmap(
        mean_corr_matrix, 
        annot=True, 
        fmt=".3f", 
        cmap="RdBu_r", # 红蓝配色，1为深红，0为白
        center=0,
        vmin=-1, vmax=1,
        xticklabels=[m.upper() for m in modalities], 
        yticklabels=[m.upper() for m in modalities],
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8, "label": "Pearson Correlation (r)"}
    )

    plt.title('Representational Similarity Analysis (MultiMAE Encoder)', fontsize=15, pad=20)
    
    # 保存结果
    save_path = os.path.join(save_dir, 'modality_correlation_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches='tight')
    plt.show()
    
    print(f"\nAverage Correlation Matrix:\n{mean_corr_matrix}")
    print(f"Correlation heatmap saved to: {save_path}")

if __name__ == '__main__':
    # 配置
    device = 'cuda:0'
    ckpt_path = "./logs/pre_train_mask_patch/ploss_0/20260501_135925/checkpoints/epoch_100_.pth"
    json_path = "./jsons/brats2018.json"
    output_dir = "./auxiliary/images/correlation_0"

    # 1. 初始化模型
    model = instantiate_MultiMAE(task="rec", mask_ratio=0.0, adapter_depth=0, encoder_depth=24)
    model = load_checkpoint(model, ckpt_path, device=device)
    model.to(device)

    # 2. 创建数据加载器
    _, val_loader = create_loader(
        task="rec", num_samples=1, split_json=json_path, 
        img_size=(160, 176, 144), batch_size=2
    )

    # 3. 执行分析
    evaluate_and_plot_correlation(
        model=model, 
        dataloader=val_loader, 
        device=device, 
        save_dir=output_dir
    )