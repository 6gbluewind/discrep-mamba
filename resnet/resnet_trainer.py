import os
import json
import torch
import torch.nn as nn
import numpy as np
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from monai.data import CacheDataset
from monai.transforms import Compose

class Trainer:
    def __init__(
        self, 
        model, 
        dataset_json, 
        save_dir,
        device="cuda:0",
        lr=1e-6,
        batch_size=2,
        max_epochs=100,
        val_interval=2,
        save_interval=4,
        train_transforms=None,
        val_transforms=None
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataset_json = dataset_json
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.val_interval = val_interval
        self.save_interval = save_interval
        
        # 创建保存目录
        self.checkpoint_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 设置日志
        self._setup_logging()
        
        # 数据转换
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

        # 优化器与损失函数
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.best_acc = -1

    def _setup_logging(self):
        log_path = os.path.join(self.save_dir, "train.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"训练环境初始化完成。模型: {type(self.model).__name__}")

    def _compute_accuracy(self, y_pred, y_true):
        _, y_pred_indices = torch.max(y_pred, dim=1)
        correct = (y_pred_indices == y_true).sum().item()
        return correct, y_true.size(0)

    def prepare_data(self):
        with open(self.dataset_json, "r") as f:
            data_list = json.load(f)

        self.logger.info("正在加载缓存数据集...")
        train_ds = CacheDataset(data=data_list["train"], transform=self.train_transforms, cache_rate=1.0)
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        val_ds = CacheDataset(data=data_list["val"], transform=self.val_transforms, cache_rate=1.0)
        self.val_loader = DataLoader(val_ds, batch_size=self.batch_size)
        self.logger.info(f"数据准备就绪。训练集: {len(train_ds)}, 验证集: {len(val_ds)}")

    def train(self):
        self.prepare_data()
        
        for epoch in range(self.max_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.max_epochs}")
            self.model.train()
            
            epoch_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}", leave=True)
            for batch_data in pbar:
                # 假设键名为 "images" 和 "label"，可根据需要通过参数扩展
                inputs = batch_data["images"].to(self.device)
                labels = batch_data["label"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                c, t = self._compute_accuracy(outputs, labels)
                train_correct += c
                train_total += t
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{train_correct/train_total:.4f}"})

            avg_loss = epoch_loss / len(self.train_loader)
            train_acc = train_correct / train_total
            self.logger.info(f"Train Result - Loss: {avg_loss:.4f}, Acc: {train_acc:.4f}")

            # 保存定期 Checkpoint
            if (epoch + 1) % self.save_interval == 0:
                ckpt_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch+1}.pth")
                torch.save(self.model.state_dict(), ckpt_path)
                self.logger.info(f"保存 Checkpoint: {ckpt_path}")

            # 验证逻辑
            if (epoch + 1) % self.val_interval == 0:
                self.run_validation()

        self.logger.info(f"训练结束。最佳验证准确率: {self.best_acc:.4f}")

    def run_validation(self):
        self.model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for val_data in self.val_loader:
                v_inputs = val_data["images"].to(self.device)
                v_labels = val_data["label"].to(self.device)
                v_outputs = self.model(v_inputs)
                
                c, t = self._compute_accuracy(v_outputs, v_labels)
                val_correct += c
                val_total += t
            
            val_acc = val_correct / val_total
            self.logger.info(f"Validation Result - Accuracy: {val_acc:.4f}")
            
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                best_model_path = os.path.join(self.save_dir, "best_model.pth")
                torch.save(self.model.state_dict(), best_model_path)
                self.logger.info(f">>> 发现新最佳模型! 已保存至: {best_model_path}")