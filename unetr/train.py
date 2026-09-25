import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch
import argparse
import matplotlib

from torch import nn
from tqdm import tqdm
from torch.optim import AdamW
from unetr.unetr import UNETR
from matplotlib import pyplot as plt
from trainer.trainer import Trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data.create_datasets import get_loader
from torchmetrics.segmentation import DiceScore, MeanIoU
from monai.losses import DiceLoss
from data.data_transforms import MultiLabelsToBraTSClasses

matplotlib.use("Agg")

parser = argparse.ArgumentParser('UNETR training')

parser.add_argument("--in_channels", type=int, default=1)
parser.add_argument("--out_channels",  type=int, default=3)
parser.add_argument("--img_size", type=tuple, default=(128,128,128))
parser.add_argument("--task", type=str, default='seg')
parser.add_argument("--num_samples", type=int, default=1)
parser.add_argument("--split_json", type=str, default='./jsons/brats2023_gli.json')

parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--num_epochs", type=int, default=300)
parser.add_argument("--val_interval", type=int, default=5)
parser.add_argument("--save_interval", type=int, default=20)
parser.add_argument("--log_dir", type=str, default='./logs/unetr_train')
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--clip_grad_val", type=float, default=0.5)

parser.add_argument("--input_tasks", type=list, default=['t1c'])

args = parser.parse_args()


def creat_unetr(in_channels=1, out_channels=3, img_size=(128,128,128), proj_type='conv', norm_name='instance'):
    net = UNETR(in_channels=in_channels, out_channels=out_channels,
                img_size=img_size, proj_type=proj_type, norm_name=norm_name)
    
    return net

def create_loader(task='seg', num_samples=1, split_json='./jsons/brast2023_gli.json', img_size=(128,128,128), batch_size=2):
    train_loader = get_loader(task=task, num_samples=num_samples, is_train=True, split_json=split_json, img_size=img_size, batch_size=batch_size)
    val_loader = get_loader(task=task, num_samples=num_samples, is_train=False, split_json=split_json, img_size=img_size, batch_size=batch_size)

    return train_loader, val_loader
    

class UnetrTrainer(Trainer):
    def __init__(self, model, device, train_loader, val_loader, num_epochs, val_interval, save_interval, log_dir='./logs', 
                lr=1e-4, weight_decay=0.1, test_stage=False, num_seg_class=3, multilabel=True, clip_grad_val=0.5,
                *args, **kwargs):
        super().__init__(model, device, train_loader, val_loader, num_epochs, val_interval, save_interval, log_dir, *args, **kwargs)
        self.optimizer = AdamW(self.model.parameters(),lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10)
        self.loss = DiceLoss(include_background=True, to_onehot_y=False, softmax=False, sigmoid=True)
        
        self.test_stage = test_stage
        self.num_seg_classes = num_seg_class
        self.multilabel = multilabel
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.clip_grad_val = clip_grad_val
        self.label_transformer = MultiLabelsToBraTSClasses()

        self.metrics = nn.ModuleDict(
            {
                "dice": DiceScore(
                    num_classes=self.num_seg_classes,
                    average=None if self.test_stage else "macro",
                    input_format="one-hot" if self.multilabel else "index",
                ),
                "mIoU": MeanIoU(
                    num_classes=self.num_seg_classes,
                    input_format="one-hot" if self.multilabel else "index",
                ),
            }
        ).to(self.device)

    def calcualte_loss(self, target, output):
        assert target.shape == output.shape

        return self.loss(output, target)

    def calculate_metrics(self, target, output):
        metric_results = {}
        target = target.long()
        output = (output.sigmoid() > 0.5 if self.multilabel else output.argmax(dim=1, keepdim=True))
        
        for name, metric in self.metrics.items():
            metric_result = metric(output, target)

            if metric_result.dim() == 0:
                metric_results[name] = metric_result
            else:
                for i in range(metric_result.shape[0]):
                    metric_results[f"{name}-{i}"] = metric_result[i]

        if self.test_stage:
            for sample_idx in range(output.shape[0]):
                sample_values = {}
                for name, metric in self.metrics.items():
                    metric_result = (metric(output[sample_idx][None, ...], target[sample_idx][None, ...]).detach().cpu())
                    if metric_result.dim() == 0:
                        sample_values[name] = metric_result.item()
                    else:
                        for i in range(metric_result.shape[0]):
                            sample_values[f"{name}-{i}"] = metric_result[i].item()
                self.test_results.append(sample_values)

        return metric_results

    @torch.no_grad
    def validation(self, epoch, input_tasks):
        self.model.eval()
        epoch_metrics = {}
        pbar = tqdm(self.val_loader, desc=f"{epoch}/{self.num_epochs}")
        num_batches = len(self.val_loader) // 2

        for step, batch in enumerate(pbar):
            input = []
            for task in input_tasks:
                input.append(batch[task])

            input = self._move_to_device(torch.cat(input, dim=1))
            target = self._move_to_device(batch['seg'])
            output = self.model(input)

            batch_metrics = self.calculate_metrics(target, output)
            for name, val in batch_metrics.items():
                if name not in epoch_metrics:
                    epoch_metrics[name] = []
                epoch_metrics[name].append(val)

            if (step + 1) == (num_batches):
                self.visualize(input, output, target, epoch)
            
        for metric_key, metric_values in epoch_metrics.items():
            avg_metric = torch.stack(metric_values).mean()
            self.writer.add_scalar(f'val/{metric_key}', avg_metric, epoch)
            self.logger.info(f'{metric_key} : {avg_metric:.4f}')

    def train_epoch(self, epoch, input_tasks):
        epoch_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"{epoch}/{self.num_epochs}")
        for batch in pbar:
            input = []
            for task in input_tasks:
                input.append(batch[task])

            input = self._move_to_device(torch.cat(input, dim=1))
            target = self._move_to_device(batch['seg'])

            self.optimizer.zero_grad()
            output = self.model(input)
            
            batch_loss = self.calcualte_loss(target, output)
            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_val)
            self.optimizer.step()
            self.optimizer.zero_grad()

            epoch_loss += batch_loss.item()

            pbar.set_postfix({
                'loss': batch_loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        avg_batch_loss = epoch_loss / len(self.train_loader)
        self.writer.add_scalar('train/loss', avg_batch_loss, epoch)
        self.writer.add_scalar( 'lr', self.optimizer.param_groups[0]['lr'], epoch)
        self.logger.info(f'Epoch {epoch} : average loss {avg_batch_loss:.4f}')

        self.scheduler.step(avg_batch_loss)
    
    def visualize(self, input, output, target, epoch, batch_idx=0):
        fig, axes = plt.subplots(1, 3, figsize=(18, 10))

        input_ax = axes[0]
        input_slice = input.shape[-1] // 2
        input_ax.imshow(input[batch_idx, 0, ..., input_slice].detach().cpu(), cmap='gray')
        input_ax.set_title('Input')
        input_ax.axis('off')

         
        if self.multilabel:
            output = (output.sigmoid() > 0.5).float()
        else:
            output = (output.argmax(dim=1, keepdim=True).float())
        output = self.label_transformer(output)
        output_ax = axes[1]
        output_slice = output.shape[-1] // 2
        output_ax.imshow(output[batch_idx, 0, ..., output_slice].detach().cpu(), cmap='viridis', vmin=0, vmax=3)
        output_ax.set_title('Output')
        output_ax.axis('off')
        
        target = self.label_transformer(target)
        target_ax = axes[2]
        target_slice = target.shape[-1] // 2
        target_ax.imshow(target[batch_idx, 0, ..., target_slice].detach().cpu(), cmap='viridis', vmin=0, vmax=3)
        target_ax.set_title('Target')
        target_ax.axis('off')

        self.writer.add_figure('val/visualize', fig, epoch)
        fig_name = f"epoch_{epoch}_.png"
        fig_path = os.path.join(self.image_path, fig_name)
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')

        plt.close(fig)


    def train(self, input_tasks=['t1']):
        self.logger.info(f"input tasks: {input_tasks}")
        for epoch in range(1, self.num_epochs+1):
            self.model.train()
            self.train_epoch(epoch, input_tasks=input_tasks)

            if epoch % self.val_interval == 0:
                self.validation(epoch, input_tasks)

            if epoch % self.save_interval == 0:
                self.save_checkpoint(epoch)

def main(args):
    model = creat_unetr(in_channels=args.in_channels, out_channels=args.out_channels, img_size=args.img_size)
    train_loader, val_loader = create_loader(
        task=args.task, num_samples=args.num_samples, split_json=args.split_json,
        img_size=args.img_size, batch_size=args.batch_size
    )
    
    trainer = UnetrTrainer(
        model=model, device=args.device, train_loader=train_loader, val_loader=val_loader,
        num_epochs=args.num_epochs, val_interval=args.val_interval, save_interval=args.save_interval,
        log_dir=args.log_dir, lr=args.lr, weight_decay=args.weight_decay, test_stage=False, 
        num_seg_class=args.out_channels, multilabel=True, clip_grad_val=args.clip_grad_val
    )

    trainer.train(input_tasks=args.input_tasks)


if __name__ == '__main__':
    main(args)      
    
