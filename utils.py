import os
import torch
import datetime
import logging
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from data.create_datasets import get_loader

def get_summarized_keys(keys, depth=2):
    summarized = set()
    for key in keys:
        parts = key.split('.')
        if 'embed' in key and len(parts) > depth:
            current_depth = depth + 1 
        else:
            current_depth = depth
        summarized.add(".".join(parts[:current_depth]))
    return summarized

def load_checkpoint(model, checkpoint_path, device, keys_remove=[],):
        model = model.to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_state = checkpoint['model_state_dict']
        filted_model_state = {}
        
        for k, v in model_state.items():
            if k.split(".")[0] not in keys_remove:
                filted_model_state[k]=v
        missing_keys, unexpected_keys = model.load_state_dict(filted_model_state, strict=False)

        # missing_keys = set([key.split(".")[0]+"."+key.split(".")[1] for key in missing_keys])
        # unexpected_keys = set([key.split(".")[0]+"."+key.split(".")[1] for key in unexpected_keys])
        missing_keys = get_summarized_keys(missing_keys, depth=3)
        unexpected_keys = get_summarized_keys(unexpected_keys, depth=3)
        
        print(f"missing keys: {len(missing_keys)}, {missing_keys}")
        print(f"unexpected keys: {len(unexpected_keys)}, {unexpected_keys}")
        print(f"load checkpoint from : {checkpoint_path}")
        return model


def freeze_weights(model, freeze_parts=None): 
    for part in freeze_parts: 
        if part == "input_adapters": 
            for param in model.input_adapters.parameters(): 
                param.requires_grad = False
            print(f"freeze input_adapters")

        if part == "embed": 
            model.embed.requires_grad = False
            print(f"freeze embed")
        
        if part == "encoder": 
            for param in model.encoder.parameters(): 
                param.requires_grad = False
            print(f"freeze encoder")

        if part == "output_adapters": 
            for param in model.output_adapters.parameters(): 
                param.requires_grad = False
            print(f"freeze output_adapters")

    return model



def create_loader(task='seg', num_samples=1, split_json='./jsons/brast2023_gli.json', img_size=(128,128,128), batch_size=2, modals=["t1n","t1c","t2w","t2f"]):
    train_loader = get_loader(task=task, num_samples=num_samples, is_train=True, split_json=split_json, img_size=img_size, batch_size=batch_size,modals=modals)
    val_loader = get_loader(task=task, num_samples=num_samples, is_train=False, split_json=split_json, img_size=img_size, batch_size=batch_size,modals=modals)

    return train_loader, val_loader


def get_permutation(input_tasks):
        num_input_tasks = len(input_tasks)
        permutations = []
        for num_mask_modalitys in range(1, num_input_tasks):
            permutations += list(itertools.combinations(input_tasks, num_mask_modalitys))
       
        return permutations

def get_mask_ratios(input_tasks, permutations):
    mask_ratios_list = []
    for mask_modalitys in permutations:
        mask_ratios = {
            task : 1.0 if task in mask_modalitys else 0.0
            for task in input_tasks
        }
        mask_ratios_list.append(mask_ratios)

    return mask_ratios_list


def move_to_device(batch, device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, (list, tuple)):
            return [move_to_device(item, device) for item in batch]
        elif isinstance(batch, dict):
            return {key: move_to_device(value, device) for key, value in batch.items()}
        else:
            return batch

def create_log_dir(base_path, prefix=None):
    """create timestamp dir for log"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if prefix:
        folder_name = f"{prefix}_{timestamp}"
    else:
        folder_name = timestamp
    full_path = os.path.join(base_path, folder_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path


def create_logger(logging_dir='.'):
    """Create a logger that writes to a log file and stdout."""

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)

    return logger

def create_dir(parent_path, dir_name):
    """ create dir (parent_path/dir_name) """
    dir_path = os.path.join(parent_path, dir_name)
    os.makedirs(dir_path, exist_ok=True)
    
    return dir_path


def plot_rsa_matrix(matrix, labels, save_path, title=None, cmap="coolwarm"):

    plt.figure(figsize=(8, 8))
    sns.set_theme(style="white")
    
    ax = sns.heatmap(
        matrix, 
        annot=True, 
        fmt=".3f", 
        cmap=cmap,
        center=0,
        vmin=-1, vmax=1,
        xticklabels=labels, 
        yticklabels=labels,
        square=True,
        linewidths=1.5,  
        annot_kws={
            "size": 14,
            "weight": "bold" 
        },
        cbar_kws={"shrink": .75}
    )

    cbar = ax.collections[0].colorbar
    cbar.set_label('Pearson Correlation (r)', 
                   size=14, 
                   fontweight='bold', 
                   labelpad=15)
    
    cbar.ax.tick_params(labelsize=12)
    for t in cbar.ax.get_yticklabels():
        t.set_fontweight('bold')

    plt.xticks(fontsize=12, fontweight='bold', rotation=0)
    plt.yticks(fontsize=12, fontweight='bold', rotation=0)

    if title:
        plt.title(title, fontsize=15, fontweight='bold', pad=25)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"RSA Matrix saved to: {save_path}")

