import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch
import argparse
import logging
import matplotlib 

from torch import nn
from tqdm import tqdm
from unetr.unetr import UNETR
from matplotlib import pyplot as plt
from data.create_datasets import get_loader
from torchmetrics.segmentation import DiceScore, MeanIoU
from data.data_transforms import MultiLabelsToBraTSClasses
from trainer.utils import create_logger, create_log_dir, create_dir

matplotlib.use("Agg")

parser = argparse.ArgumentParser("UNETR Validation")

parser.add_argument("--in_channels", type=int, default=1)
parser.add_argument("--out_channels",  type=int, default=3)
parser.add_argument("--img_size", type=tuple, default=(128,128,128))
parser.add_argument("--task", type=str, default='seg')
parser.add_argument("--num_samples", type=int, default=1)
parser.add_argument("--split_json", type=str, default='./jsons/brats2023_gli.json')

parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--batch_size", type=int, default=2)

parser.add_argument("--log_dir", type=str, default='./logs/unetr_train')
parser.add_argument("--input_tasks", type=list, default=['t1c'])
parser.add_argument("--weight_path", type=str, default=None)

parser.add_argument("--num_seg_classes", type=int, default=3)
parser.add_argument("--test_stage", type=bool, default=True)
parser.add_argument("--multilabel", type=bool, default=True)


args = parser.parse_args()

def creat_unetr(in_channels=1, out_channels=3, img_size=(128,128,128), proj_type='conv', norm_name='instance'):
    net = UNETR(in_channels=in_channels, out_channels=out_channels,
                img_size=img_size, proj_type=proj_type, norm_name=norm_name)
    
    return net

def create_loader(task='seg', num_samples=1, split_json='./jsons/brast2023_gli.json', img_size=(128,128,128), batch_size=2):
    val_loader = get_loader(task=task, num_samples=num_samples, is_train=False, split_json=split_json, img_size=img_size, batch_size=batch_size)

    return val_loader

def load_weight(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    model_state = checkpoint['model_state_dict']
    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    print(f"Successfully load weight from {path}")

    return model

def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (list, tuple)):
        return [move_to_device(item) for item in batch]
    elif isinstance(batch, dict):
        return {key: move_to_device(value) for key, value in batch.items()}
    else:
        return batch


def calculate_metrics(metrics, target, output, multilabel):
    metric_results = {}
    target = target.long()
    output = (output.sigmoid() > 0.5 if multilabel else output.argmax(dim=1, keepdim=True))
    
    for name, metric in metrics.items():
        metric_result = metric(output, target)

        if metric_result.dim() == 0:
            metric_results[name] = metric_result
        else:
            for i in range(metric_result.shape[0]):
                metric_results[f"{name}-{i}"] = metric_result[i]

    return metric_results


def visualize(input, output, target, batch, multilabel, image_path, batch_idx=0):
    label_transformer = MultiLabelsToBraTSClasses()
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))

    input_ax = axes[0]
    input_slice = input.shape[-1] // 2
    input_ax.imshow(input[batch_idx, 0, ..., input_slice].detach().cpu(), cmap='gray')
    input_ax.set_title('Input')
    input_ax.axis('off')

        
    if multilabel:
        output = (output.sigmoid() > 0.5).float()
    else:
        output = (output.argmax(dim=1, keepdim=True).float())
    output = label_transformer(output)
    output_ax = axes[1]
    output_slice = output.shape[-1] // 2
    output_ax.imshow(output[batch_idx, 0, ..., output_slice].detach().cpu(), cmap='viridis', vmin=0, vmax=3)
    output_ax.set_title('Output')
    output_ax.axis('off')
    
    target = label_transformer(target)
    target_ax = axes[2]
    target_slice = target.shape[-1] // 2
    target_ax.imshow(target[batch_idx, 0, ..., target_slice].detach().cpu(), cmap='viridis', vmin=0, vmax=3)
    target_ax.set_title('Target')
    target_ax.axis('off')

    fig_name = f"batch_{batch}_.png"
    fig_path = os.path.join(image_path, fig_name)
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')

    plt.close(fig)


@torch.no_grad()
def main(args):
    device = args.device
    input_tasks = args.input_tasks
    log_dir = create_log_dir(args.log_dir)
    logger = create_logger(log_dir)
    image_path = create_dir(logger, 'images')

    logger.info(f"Validation run in {device}, model input tasks: {input_tasks}")

    model = creat_unetr(in_channels=args.in_channels, out_channels=args.out_channels, img_size=args.img_size).to(device)
    
    val_loader = create_loader(
        task=args.task, num_samples=args.num_samples, split_json=args.split_json, 
        img_size=args.img_size, batch_size=args.batch_size
    )

    model = load_weight(model, args.weight_path, device)

    metrics = nn.ModuleDict(
        {
            "dice": DiceScore(
                num_classes=args.num_seg_classes,
                average=None if args.test_stage else "macro",
                input_format="one-hot" if args.multilabel else "index",
            ),
            "mIoU": MeanIoU(
                num_classes=args.num_seg_classes,
                input_format="one-hot" if args.multilabel else "index",
            )
        }
    ).to(device)

    pbar = tqdm(val_loader)
    metrics_results = {}
    visualize_batch_dix = len(val_loader) // 2
    for batch_idx, batch in pbar:
        input = []
        for task in input_tasks:
            input.append(batch[task])
        input = move_to_device(torch.cat(input, dim=1), device)
        target = move_to_device(batch['seg'], device=device)
        output = model(input)

        batch_metrics = calculate_metrics(metrics, target, output, args.multilabel)
        for name, value in batch_metrics.items():
            if name not in metrics_results:
                metrics_results[name] = []
            metrics_results.append(value)
        
        if (batch_idx+1) == visualize_batch_dix:
            visualize(input, output, target, batch_idx+1, args.multilabel, image_path=image_path)

    for metric_key, metric_values in metrics_results.items():
        avg_metric = torch.stack(metric_values).mean()
        logger.info(f"{metric_key}: {avg_metric:.4f}")

if __name__ == '__main__':
    main(args)

