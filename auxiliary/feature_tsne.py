import os
import sys



parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from utils import load_checkpoint, create_loader
from models.multimae3d import instantiate_MultiMAE

@torch.no_grad()
def evaluate_and_plot_tsne(model, dataloader, device, save_dir="./logs/tsne"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    all_features = []
    all_labels = []

    modalities = ['t1n', 't1c', 't2w', 't2f']
    color_map = {'t1n': '#1f77b4', 't1c': '#ff7f0e', 't2w': '#2ca02c', 't2f': '#d62728'}

    pbar = tqdm(dataloader, desc="Extracting features for t-SNE")
    for batch in pbar:
        if isinstance(batch, dict):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        output = model(batch, mask_ratio=0.0, return_image=False)

        last_layer_tokens = output["encoder_tokens"][-1]
        task_ranges = output["task_ranges"]

        for task in modalities:
            if task in task_ranges:
                start, end = task_ranges[task]
                
                task_tokens = last_layer_tokens[:, start:end, :]
                pooled_features = task_tokens.mean(dim=1)

                all_features.append(pooled_features.cpu().numpy())
                all_labels.extend([task] * pooled_features.shape[0])

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.array(all_labels)

    print(f"\nRunning t-SNE on {all_features.shape[0]} samples with dimension {all_features.shape[1]}...")
    
    tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=30,learning_rate='auto')
    tsne_results = tsne.fit_transform(all_features)

    plt.figure(figsize=(10, 8))
    for task in modalities:
        idx = all_labels == task
        plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], 
                    label=task.upper(), c=color_map[task], alpha=0.8, edgecolors='w', s=60)

    plt.title('t-SNE Visualization of Encoded Modality Features', fontsize=15)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(title="Modalities", title_fontsize='13', fontsize='11')
    plt.grid(True, linestyle='--', alpha=0.5)

    save_path = os.path.join(save_dir, 'modality_tsne_clustering.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE visualization successfully saved to: {save_path}")

if __name__ == '__main__':
    device = 'cuda:0'
    model = instantiate_MultiMAE(task="rec", mask_ratio=0.0, adapter_depth=0, encoder_depth=24)
    model = load_checkpoint(model,"./logs/pre_train_mask_patch/ploss_0/20260501_135925/checkpoints/epoch_100_.pth", device=device)
    model.to(device)

    _, val_loader = create_loader(
        task="rec", num_samples=1, split_json="./jsons/brats2023_gli.json", 
        img_size=(160,176,144), batch_size=2
    )

    evaluate_and_plot_tsne(model=model, dataloader=val_loader, device=device, save_dir="./auxiliary/images_0")
