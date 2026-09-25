import os
import torch
import argparse
import matplotlib
import matplotlib.pyplot as plt


from tqdm import tqdm
from loss import get_loss
from torch.optim import AdamW
from trainer.trainer import Trainer
from trainer import Tasks, get_metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.multimae3d import instantiate_MultiMAE
from utils import load_checkpoint, freeze_weights, create_loader

matplotlib.use("Agg")

parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, default=None)

parser.add_argument('--img_size', nargs=3, type=int, default=[160, 176, 144])
parser.add_argument('--patch_size', nargs=3, type=int, default=[16, 16, 16])
parser.add_argument('--in_channels', type=int, default=1)
parser.add_argument('--input_tasks', nargs='+', type=str, default=['t1n', 't1c', 't2w', 't2f'])
parser.add_argument('--encoder_embed_dim', type=int, default=768)
parser.add_argument('--encoder_depth', type=int, default=24)
parser.add_argument('--adapter_depth', type=int, default=12)
parser.add_argument('--seg_classes', type=int, default=3)
parser.add_argument('--output_tasks', nargs='+', type=str, default=['t1n', 't1c', 't2w', 't2f'])
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--mask_ratio', type=float, default=0.75)
parser.add_argument('--mask_mode', type=str, default='dirichlet')

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--log_dir', type=str, default='./logs/pre_train')

parser.add_argument('--checkpoint_path', type=str, default=None)
parser.add_argument('--keys_remove', nargs='+', type=str, default=[])
parser.add_argument('--freeze_parts', nargs='+', type=str, default=[])
parser.add_argument('--num_samples', type=int, default=1)

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--split_json', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_epochs', type=int, default=700)
parser.add_argument('--val_interval', type=int, default=10)
parser.add_argument('--save_interval', type=int, default=20)

parser.add_argument('--test_stage', action='store_true')
parser.add_argument('--classification_task', type=str, default="binary")
parser.add_argument('--multilabel', action='store_true')

parser.add_argument('--diffloss', action='store_true')
parser.add_argument('--flatten', action='store_true')
parser.add_argument('--middle_token', action='store_true')

args = parser.parse_args()


class MAETrainer(Trainer):
    def __init__(
        self, task, input_tasks, output_tasks, model, device, train_loader, val_loader, num_epochs, val_interval, save_interval,
        optimizer, scheduler, metrics, loss, test_stage=False, classification_task="binary", multilabel=True, log_dir='./logs', *args, **kwargs
    ):
        super().__init__(model, device, train_loader, val_loader, num_epochs, val_interval, 
                        save_interval, optimizer, scheduler, metrics, loss, log_dir, *args, **kwargs)
        self.task = task
        self.input_tasks = input_tasks
        self.output_tasks = output_tasks
        self.multilabel = multilabel
        self.test_stage = test_stage
        self.classification_task = classification_task
        self.metric_preds = []
        self.metric_targets = []

    def calcualte_loss(self, batch, output):
        loss = self.loss(batch, output, self.input_tasks, self.output_tasks, )
        return loss

    def calculate_metrics(self, batch, output, test_stage=None):
        test_stage = self.test_stage if test_stage is None else test_stage
        metric_results = {}

        if self.task == Tasks.REC.value:
            for task in self.output_tasks:
                pred = output["reconstructed_patches"][task]
                target = batch[task]
                for name, metric in self.metrics.items():
                    if target.device != metric.device:
                        metric = metric.to(target.device)

                    metric_results[f"{task}-{name}"] = metric(pred, target)
            
            return metric_results
        
        elif self.task == Tasks.SEG.value:
            for task in self.output_tasks:
                pred = output["reconstructed_patches"][task]
                target = batch[task].long()
                metric_results = {}
                pred = (
                    pred.sigmoid() > 0.5
                    if self.multilabel
                    else pred.argmax(dim=1, keepdim=True)
                )

                for name, metric in self.metrics.items():
                    if target.device != metric.device:
                        metric = metric.to(target.device)
                    metric_result = metric(pred, target)
                    if metric_result.dim() == 0:
                        metric_results[name] = metric_result
                    else:
                        for i in range(metric_result.shape[0]):
                            metric_results[f"{name}-{i}"] = metric_result[i]

            return metric_results

        elif self.task == Tasks.CLA.value:
            for task in self.output_tasks:
                pred = output["reconstructed_patches"][task].squeeze(dim=1)
                if self.classification_task == "binary":
                    pred = pred.sigmoid()
                else:
                    pred = pred.softmax(dim=1)
                target = batch[task].long()
                self.metric_preds.append(pred.detach())
                self.metric_targets.append(target.detach())

                if self.test_stage:
                    for sample_idx in range(target.shape[0]):
                        sample_values = {}
                        if self.classification_task == "binary":
                            sample_values["pred"] = ((pred[sample_idx].detach().cpu() > 0.5).int().item())
                        else:
                            sample_values["pred"] = torch.argmax(pred[sample_idx].detach().cpu()).item()
                        sample_values["target"] = target[sample_idx].detach().cpu().item()
                        self.test_results.append(sample_values)
            return {}
    
    def _comp_metrics(self, pred, target) :
        metric_results = {}
        for name, metric in self.metrics.items():
            if target.device != metric.device:
                metric = metric.to(target.device)
            metric_results[name] = metric(pred, target)
        return metric_results
    
    def epoch_end_comp_metrics(self):
        pred = torch.cat(self.metric_preds, dim=0)
        n_samples = pred.shape[0]
        target = torch.cat(self.metric_targets, dim=0)
        pred = pred.as_tensor() if hasattr(pred, "as_tensor") else pred
        target = target.as_tensor() if hasattr(target, "as_tensor") else target
        metric_results = self._comp_metrics(pred, target)
        self.metric_preds = []
        self.metric_targets = []
        return metric_results, n_samples

    def inference(self, batch):
        output = self.model(batch)
        return output
    
    def train_epoch(self, epoch):
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}")
        loss = 0.0
        loss_val = {}
        num_batches = len(self.train_loader)

        for batch in pbar:
            batch = self._move_to_device(batch)
            self.optimizer.zero_grad()

            output = self.inference(batch)
            batch_loss, batch_loss_val = self.loss(batch, output, self.output_tasks)
            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss += batch_loss.item()
            for task, value in batch_loss_val.items():
                if task not in loss_val:
                    loss_val[task] = 0.0
                loss_val[task] += value
            
            pbar.set_postfix({
                'loss': batch_loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            

        # Average loss
        loss /= num_batches
        for task, value in loss_val.items():
            loss_val[task] = value / num_batches
            self.writer.add_scalar(f'train/loss/{task}', loss_val[task], epoch)
            
        self.writer.add_scalar('train/loss', loss, epoch)
        self.logger.info(f'Epoch {epoch} : average loss {loss:.4f}')

        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(loss)
            else:
                self.scheduler.step()

        self.writer.add_scalar( 'lr', self.optimizer.param_groups[0]['lr'], epoch)

    @torch.no_grad
    def validation(self, epoch):
        self.model.eval()
        metrics = {}
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.num_epochs}")    
        num_batches = len(self.val_loader)

        for step, batch in enumerate(pbar):
            batch = self._move_to_device(batch)
            output = self.model(batch)
            batch_metrics = self.calculate_metrics(batch, output)

            for name, val in batch_metrics.items():
                if name not in metrics:
                    metrics[name] = []
                metrics[name].append(val)

            if (step + 1) == (1) and self.task != Tasks.CLA.value:
                self.visualize(batch, output, epoch)
            
        for metric_key, metric_values in metrics.items():
            avg_metric = torch.stack(metric_values).mean()
            self.writer.add_scalar(f'val/{metric_key}', avg_metric, epoch)
            self.logger.info(f'{metric_key} : {avg_metric:.4f}')

        if self.task == Tasks.CLA.value:
            metric_results, num_smaples = self.epoch_end_comp_metrics()
            self.logger.info(f"num samples: {num_smaples}")
            for metric_key, metric_value in metric_results.items():
                self.logger.info(f'{metric_key} : {metric_value:.4f}')

    def train(self):
        for epoch in range(1, self.num_epochs+1):
            self.model.train()
            self.train_epoch(epoch)

            if epoch % self.val_interval == 0:
                self.validation(epoch)

            if epoch % self.save_interval == 0:
                self.save_checkpoint(epoch)
    
    def visualize(self, batch, output, epoch, cmap='gray'):
        rows = len(self.input_tasks)
        fig, axes = plt.subplots(rows, 3, figsize=(18, 10))
        input_tasks = [task for task in self.input_tasks]

        for i, task in enumerate(input_tasks):
            input_ax = axes[i][0]
            slice_idx = output['selected_patches'][task].shape[-1] // 2
            input_ax.imshow(output['selected_patches'][task][0, 0, ..., slice_idx].detach().cpu(), cmap=cmap)
            input_ax.set_title(f'Input-{task}')
            input_ax.axis('off')

        for i, task in enumerate(self.output_tasks):
            gt_ax = axes[i][1]
            slice_idx = batch[task].shape[-1] // 2
            gt_ax.imshow(batch[task][0, 0, ..., slice_idx].detach().cpu(), cmap=cmap)
            gt_ax.set_title(f'Ground Truth-{task}')
            gt_ax.axis('off')

            output_ax = axes[i][2]
            slice_idx = output['reconstructed_patches'][task].shape[-1] // 2
            output_ax.imshow(output['reconstructed_patches'][task][0, 0, ..., slice_idx].detach().cpu(), cmap=cmap)
            output_ax.set_title(f'Output-{task}')
            output_ax.axis('off')

        self.writer.add_figure('val/visualize', fig, epoch)
        fig_name = f"epoch_{epoch}_.png"
        fig_path = os.path.join(self.image_path, fig_name)
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')

        plt.close(fig)

if __name__ == '__main__':
    middle_token = args.middle_token 
    model = instantiate_MultiMAE(task=args.task, img_size=tuple(args.img_size), patch_size=tuple(args.patch_size), in_channels=args.in_channels,
                                input_tasks=args.input_tasks, encoder_embed_dim=args.encoder_embed_dim,
                                encoder_depth=args.encoder_depth, adapter_depth=args.adapter_depth,
                                seg_classes=args.seg_classes, output_tasks=args.output_tasks, num_classes=args.num_classes,
                                mask_ratio=args.mask_ratio, mask_mode=args.mask_mode,middle_token=middle_token)
    
    if args.checkpoint_path is not None:
        model = load_checkpoint(model, args.checkpoint_path, device=args.device, keys_remove=args.keys_remove)
    
    if args.freeze_parts:
        model = freeze_weights(model, args.freeze_parts)

    train_loader, val_loader = create_loader(task=args.task, num_samples=args.num_samples, split_json=args.split_json, 
                                img_size=tuple(args.img_size), batch_size=args.batch_size,modals=args.input_tasks)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    
    metrics = get_metrics(task=args.task, test_stage=args.test_stage, num_seg_classes=args.seg_classes, multilabel=args.multilabel,
                          classification_task=args.classification_task, num_classes=args.num_classes)
    
    loss = get_loss(task=args.task, device=args.device, num_classes=args.num_classes, 
                    in_channels=args.in_channels, diffloss=args.diffloss)
    
    trainer = MAETrainer(task=args.task, input_tasks=args.input_tasks, output_tasks=args.output_tasks,
                        model=model, device=args.device, train_loader=train_loader, val_loader=val_loader, 
                        num_epochs=args.num_epochs, val_interval=args.val_interval, save_interval=args.save_interval,
                        optimizer=optimizer, scheduler=scheduler, metrics=metrics, loss=loss, 
                        test_stage=args.test_stage, classification_task=args.classification_task, multilabel=args.multilabel, log_dir=args.log_dir)
    
    trainer.train()
       