import os
import sys
import torch
import logging
import matplotlib

from torch import nn
from matplotlib import pyplot as plt
from utils import create_log_dir, create_logger, create_dir
from torch.utils.tensorboard import SummaryWriter

matplotlib.use('Agg')

class Trainer:
    def __init__(
        self, model, device, train_loader, val_loader ,
        num_epochs, val_interval, save_interval, optimizer, scheduler, metrics, loss, log_dir='./logs',
        *args, **kwargs,
    ):  
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = create_log_dir(log_dir)
        self.logger = create_logger(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)

        self.device = device if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Trainer device is {self.device}")
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.val_interval = val_interval
        self.save_interval = save_interval
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = nn.ModuleDict(metrics).to(self.device)
        self.loss = loss
        
        self.checkpoint_path = create_dir(self.log_dir, 'checkpoints')
        self.image_path = create_dir(self.log_dir, 'images')       
        
        self.test_results = []

    def calculate_metrics(self):
        pass

    def calcualte_loss(self):
        pass

    def train_epoch(self):
        pass

    def train(self):
        pass

    def validation(self):
        pass

    def visualize(self):
        pass

    def _move_to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, (list, tuple)):
            return [self._move_to_device(item) for item in batch]
        elif isinstance(batch, dict):
            return {key: self._move_to_device(value) for key, value in batch.items()}
        else:
            return batch
    
    def save_checkpoint(self, epoch):
        checkponit = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
        }
        checkponit_path= os.path.join(self.checkpoint_path, f'epoch_{epoch}_.pth')
        torch.save(checkponit, checkponit_path)
        self.logger.info(f"save checkpoint to {checkponit_path}")

    # def load_checkpoint(self, checkpoint_path, keys_remove=[]):
    #     checkpoint = torch.load(checkpoint_path, map_location=self.device)
    #     model_state = checkpoint['model_state_dict']
    #     filted_model_state = {}
        
    #     for k, v in model_state.items():
    #         if k.split(".")[0] not in keys_remove:
    #             filted_model_state[k]=v
    #     missing_keys, unexpected_keys = self.model.load_state_dict(filted_model_state, strict=False)

    #     missing_keys = [key.split(".")[0] for key in missing_keys]
    #     unexpected_keys = [key.split(".")[0] for key in unexpected_keys]
        
    #     logging.info(f'missing keys: {len(missing_keys)}')
    #     logging.info(f"{missing_keys}")
    #     logging.info(f'unexpected keys: {len(unexpected_keys)}')
    #     logging.info(f"{unexpected_keys}")
    #     logging.info(f'load model weight from {checkpoint_path}')


    def load_checkpoint(self, checkpoint_path, keys_remove=[],):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_state = checkpoint['model_state_dict']
        filted_model_state = {}
        
        for k, v in model_state.items():
            if k.split(".")[0] not in keys_remove:
                filted_model_state[k]=v
        missing_keys, unexpected_keys = self.model.load_state_dict(filted_model_state, strict=False)

        missing_keys = set([key.split(".")[0]+"."+key.split(".")[1] for key in missing_keys])
        unexpected_keys = set([key.split(".")[0]+"."+key.split(".")[1] for key in unexpected_keys])
        
        self.logger.info(f"missing keys: {len(missing_keys)}, {missing_keys}")
        self.logger.info(f"unexpected keys: {len(unexpected_keys)}, {unexpected_keys}")
        self.logger.info(f"load checkpoint from : {checkpoint_path}")



    def freeze_weights(self, freeze_parts=None): 
        for part in freeze_parts: 
            if part == "input_adapters": 
                for param in self.model.input_adapters.parameters(): 
                    param.requires_grad = False
                self.logger.info(f"freeze input_adapters")

            if part == "embed": 
                self.model.embed.requires_grad = False
                self.logger.info(f"freeze embed")
            
            if part == "encoder": 
                for param in self.model.encoder.parameters(): 
                    param.requires_grad = False
                self.logger.info(f"freeze encoder")

            if part == "output_adapters": 
                for param in self.model.output_adapters.parameters(): 
                    param.requires_grad = False
                self.logger.info(f"freeze output_adapters")






