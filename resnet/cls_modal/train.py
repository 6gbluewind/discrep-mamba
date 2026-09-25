import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.networks.nets import resnet50
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    Resized, ScaleIntensityd, EnsureTyped
)
import numpy as np
import monai.transforms as tf
from monai.data import CacheDataset
from tqdm import tqdm
import logging

# --- 日志配置 ---
log_name = "./train_log.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_name),  
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("日志系统已启动，训练即将开始...")


# --- 参数配置 ---
JSON_PATH = "./brats_modality.json"
CHECKPOINT_DIR = "./checkpoints"
BATCH_SIZE = 2
MAX_EPOCHS = 100
VAL_INTERVAL = 2
SAVE_INTERVAL = 4
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_accuracy(y_pred, y_true):
    _, y_pred_indices = torch.max(y_pred, dim=1)
    correct = (y_pred_indices == y_true).sum().item()
    return correct, y_true.size(0) 


train_transforms = Compose([
    tf.LoadImaged(keys=["image"],image_only=True, ensure_channel_first=True),
    tf.Orientationd(keys=['image'], axcodes="RAS"),
    tf.EnsureTyped(keys=['image'], dtype=np.float32),
    tf.ToTensord(keys=['image']),
    tf.SpatialPadd(keys=['image'], spatial_size=(160, 176, 144), mode='constant', allow_missing_keys=True),
    tf.CenterSpatialCropd(keys=['image'], roi_size=(160, 176, 144),),
    tf.RandZoomd(keys=['image',], prob=0.2, min_zoom=1.0, max_zoom=1.4, mode=['trilinear'], num_samples=1),
    tf.RandFlipd(keys=['image'], prob=0.2, spatial_axis=[0, 1]),
    tf.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=[1.0, 2.0], retain_stats=True),
    tf.ScaleIntensityRangePercentilesd(keys=['image'], lower=0.5, upper=99.5, b_min=-1, b_max=1, clip=True, relative=False, channel_wise=True),
])

val_transforms = Compose([
    tf.LoadImaged(keys=["image"],image_only=True, ensure_channel_first=True),
    tf.Orientationd(keys=['image'], axcodes="RAS"),
    tf.EnsureTyped(keys=['image'], dtype=np.float32),
    tf.ToTensord(keys=['image']),
    tf.SpatialPadd(keys=['image'], spatial_size=(160, 176, 144), mode='constant', allow_missing_keys=True),
    tf.CenterSpatialCropd(keys=['image'], roi_size=(160, 176, 144),),
    tf.ScaleIntensityRangePercentilesd(keys=['image'], lower=0.5, upper=99.5, b_min=-1, b_max=1, clip=True, relative=False, channel_wise=True),
])

with open(JSON_PATH, "r") as f:
    data_list = json.load(f)

train_ds = CacheDataset(data=data_list["train"], transform=train_transforms, cache_rate=1.0)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

val_ds = CacheDataset(data=data_list["val"], transform=val_transforms, cache_rate=1.0)
val_loader = DataLoader(val_ds, batch_size=2)


model = resnet50(spatial_dims=3, n_input_channels=1, num_classes=4).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- 训练主循环 ---
best_acc = -1

for epoch in range(MAX_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{MAX_EPOCHS}")
    model.train()
    
    epoch_loss = 0
    train_correct = 0
    train_total = 0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{MAX_EPOCHS}", leave=True)

    for batch_data in train_pbar:
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        c, t = compute_accuracy(outputs, labels)
        train_correct += c
        train_total += t

        train_pbar.set_postfix({
            "loss": f"{loss.item():.4f}", 
            "acc": f"{train_correct/train_total:.4f}"
        })

    avg_loss = epoch_loss / len(train_loader)
    train_acc = train_correct / train_total
    logger.info(f"Train - Loss: {avg_loss:.4f}, Acc: {train_acc:.4f}")

    # save checkpoint 
    if (epoch + 1) % SAVE_INTERVAL == 0:
        save_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        logger.info(f"Checkpoint saved: {save_path}")

    # validation
    if (epoch + 1) % VAL_INTERVAL == 0:
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_data in val_loader:
                v_inputs, v_labels = val_data["image"].to(device), val_data["label"].to(device)
                v_outputs = model(v_inputs)
                
                c, t = compute_accuracy(v_outputs, v_labels)
                val_correct += c
                val_total += t
            
            val_acc = val_correct / val_total
            logger.info(f"Validation - Accuracy: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "best_model.pth")
                logger.info(f">>> New Best Model Saved! Best Acc: {best_acc:.4f}")

logger.info(f"训练任务结束。最终最佳准确率: {best_acc:.4f}")