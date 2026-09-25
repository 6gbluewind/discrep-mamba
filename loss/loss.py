import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from monai.losses import DiceLoss


class PixelLoss(nn.Module):
    def __init__(self,instance=nn.MSELoss(reduction="mean"), device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = instance

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        self.to(self.device)

    def forward(self, batch, output, output_tasks, *args, **kwargs):
        loss = 0.0
        loss_value = {}
        for task in output_tasks:
            groud_truth = batch[task]
            reconstructed = output['reconstructed_patches'][task]
            task_loss = self.loss(groud_truth, reconstructed)
            loss += task_loss
            loss_value[task] = task_loss
        
        loss /= len(output_tasks)
        for task, value in loss_value.items():
            loss_value[task] = value / len(output_tasks)

        return loss, loss_value


class SegLoss(nn.Module):
    def __init__(self, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = DiceLoss(include_background=True, to_onehot_y=False, softmax=False, sigmoid=True)

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        self.to(self.device)

    def forward(self, batch, output, output_tasks, *args, **kwargs):
        loss = 0.0
        loss_value = {}
        for task in output_tasks:
            groud_truth = batch[task]
            reconstructed = output['reconstructed_patches'][task]
            task_loss = self.loss(groud_truth, reconstructed)
            loss += task_loss
            loss_value[task] = task_loss
        
        loss /= len(output_tasks)
        for task, value in loss_value.items():
            loss_value[task] = value / len(output_tasks)

        return loss, loss_value
    

class ClassLoss(nn.Module):
    def __init__(self, num_classes, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.loss = BCEWithLogitsLoss() if int(self.num_classes) <=2 else CrossEntropyLoss()
        
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        self.to(self.device)

    def forward(self, batch, output, output_tasks, *args, **kwargs):
        loss = 0.0
        loss_value = {}
        for task in output_tasks:
            target = batch[task]
            pred = output["reconstructed_patches"][task].squeeze(dim=1)
            if target.dtype != pred.dtype:
                target = target.to(dtype=pred.dtype)
            task_loss = self.loss(pred, target)
            loss_value[task] = task_loss
            loss += task_loss

        return loss, loss_value