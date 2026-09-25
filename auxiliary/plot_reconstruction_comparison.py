import os
import sys
import torch
import matplotlib
import matplotlib.pyplot as plt


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.multimae3d import instantiate_MultiMAE
from utils import load_checkpoint, create_loader, move_to_device

matplotlib.use("Agg")

def build_model(device, checkpoint_path):
    model = instantiate_MultiMAE(
        task="rec", 
        img_size=(160, 176, 144), 
        patch_size=(16, 16, 16), 
        in_channels=1,
        input_tasks=['t1n', 't1c', 't2w', 't2f'], 
        encoder_embed_dim=768, 
        encoder_depth=24,
        adapter_depth=12, 
        seg_classes=3, 
        output_tasks=['t1n', 't1c', 't2w', 't2f'], 
        num_classes=2,
        mask_ratio=0.75, 
        mask_mode="dirichlet",
        middle_token=True
    )
    
    
    model = load_checkpoint(model, checkpoint_path, device=device, keys_remove=[])
    model = model.to(device)
    model.eval()
    
    model.mask_ratio = {'t1n': 0.0, 't1c': 0.0, 't2w': 0.0, 't2f': 0.0}
    
    return model

mapping = {
    't1n': "T1", 't1c': "T1CE", 't2w': "T2", 't2f': "FLAIR"
}


def main():
    device = "cuda:0"
    output_dir = "./logs/evaluate_reconstruct/paper_reconstruct/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    mse_checkpoint_path = "logs/pre_train_mask_patch/mdloss_6/20260419_095232/checkpoints/epoch_140_.pth" 
    proposed_checkpoint_path = "logs/pre_train_mask_patch/mdloss_6/20260419_095232/checkpoints/epoch_420_.pth"
    
    print("Loading Baseline (MSE) Model...")
    model_mse = build_model(device, mse_checkpoint_path)
    
    print("Loading Proposed (MD-Loss) Model...")
    model_proposed = build_model(device, proposed_checkpoint_path)

    _, val_loader = create_loader(
        task="rec", 
        num_samples=1, 
        split_json="./jsons/image.json", 
        img_size=(160, 176, 144), 
        batch_size=2,
        modals=['t1n', 't1c', 't2w', 't2f']
    )
    
    batch = next(iter(val_loader))
    batch = move_to_device(batch, device)


    with torch.no_grad():
        print("Running inference...")
        output_mse = model_mse(batch)
        output_proposed = model_proposed(batch)


    # ==========================================
    # 4. 绘图 (改为 5行 4列，增加误差热力图)
    # ==========================================
    tasks = ['t1n', 't1c', 't2w', 't2f']
    fig, axes = plt.subplots(5, 4, figsize=(10, 12))
    plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0.1, right=0.98, top=0.95, bottom=0.05)    
    

    for col_idx, task in enumerate(tasks):
        slice_idx = batch[task].shape[-1] // 2
        
        gt_img = batch[task][0, 0, ..., slice_idx].detach().cpu()
        mse_img = output_mse['reconstructed_patches'][task][0, 0, ..., slice_idx].detach().cpu()
        prop_img = output_proposed['reconstructed_patches'][task][0, 0, ..., slice_idx].detach().cpu()

        # --------------------------------------------------
        # 核心修改：计算绝对误差图 (Error Maps)
        # --------------------------------------------------
        err_mse = torch.abs(gt_img - mse_img)
        err_prop = torch.abs(gt_img - prop_img)

        vmax_err = max(err_mse.max().item(), err_prop.max().item())

        ax_gt = axes[0][col_idx]
        ax_gt.imshow(gt_img, cmap='gray')
        if col_idx == 0:
            ax_gt.set_ylabel("Ground Truth", fontsize=16, fontweight='bold')
        ax_gt.set_title(f"{mapping[task]}", fontsize=18)
        ax_gt.set_xticks([]); ax_gt.set_yticks([])

        # 第二行：Baseline 重建图
        ax_mse = axes[1][col_idx]
        ax_mse.imshow(mse_img, cmap='gray')
        if col_idx == 0:
            ax_mse.set_ylabel("MSE", fontsize=16, fontweight='bold')
        ax_mse.set_xticks([]); ax_mse.set_yticks([])

        # 第三行：Baseline 误差图 (使用 Reds colormap)
        ax_err_mse = axes[2][col_idx]
        ax_err_mse.imshow(err_mse, cmap='Reds', vmin=0, vmax=vmax_err)
        if col_idx == 0:
            ax_err_mse.set_ylabel("MSE Error\n(Red = Higher)", fontsize=16, fontweight='bold', color='red')
        ax_err_mse.set_xticks([]); ax_err_mse.set_yticks([])

        # 第四行：Proposed 重建图
        ax_prop = axes[3][col_idx]
        ax_prop.imshow(prop_img, cmap='gray')
        if col_idx == 0:
            ax_prop.set_ylabel("SAFC", fontsize=16, fontweight='bold')
        ax_prop.set_xticks([]); ax_prop.set_yticks([])

        # 第五行：Proposed 误差图 (使用 Reds colormap)
        ax_err_prop = axes[4][col_idx]
        ax_err_prop.imshow(err_prop, cmap='Reds', vmin=0, vmax=vmax_err)
        if col_idx == 0:
            ax_err_prop.set_ylabel("SAFC Error\n(Red = Higher)", fontsize=16, fontweight='bold', color='red')
        ax_err_prop.set_xticks([]); ax_err_prop.set_yticks([])

    save_path = os.path.join(output_dir, "reconstruction_comparison.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    print(f"Figure saved successfully at: {save_path}")

if __name__ == '__main__':
    main()