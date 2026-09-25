import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from monai.networks.nets import resnet18, resnet50
import monai.transforms as tf
import numpy as np
from resnet_trainer import Trainer


class RandBlockCombine(tf.MapTransform):
    def __init__(self, keys, block_size=8):
        super().__init__(keys)
        self.block_size = block_size

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]  # 预期 shape: (4, D, H, W)
            c, depth, height, width = img.shape
            bs = self.block_size

            # 确保尺寸能被 block_size 整除
            assert depth % bs == 0 and height % bs == 0 and width % bs == 0, \
                f"图像尺寸 {depth, height, width} 必须能被 {bs} 整除"

            # 1. 将图像切分为 blocks: (4, D//8, 8, H//8, 8, W//8, 8)
            new_shape = (c, depth // bs, bs, height // bs, bs, width // bs, bs)
            blocks = img.reshape(new_shape)

            # 2. 随机生成每个位置选择哪个通道的索引 (D//8, H//8, W//8)
            # 例如：rand_indices[0,0,0] = 2 表示第1个块取第3个模态 (T1ce)
            rand_indices = torch.randint(0, c, (depth // bs, height // bs, width // bs))

            # 3. 构造输出容器 (1, D//8, 8, H//8, 8, W//8, 8)
            # 这里的逻辑：在 C 维度根据 rand_indices 进行 gather
            output = torch.zeros((1, depth // bs, bs, height // bs, bs, width // bs, bs), device=img.device)
            
            for i in range(depth // bs):
                for j in range(height // bs):
                    for k in range(width // bs):
                        idx = rand_indices[i, j, k]
                        output[0, i, :, j, :, k, :] = blocks[idx, i, :, j, :, k, :]

            # 4. 还原回原始空间形状 (1, D, H, W)
            d[key] = output.reshape(1, depth, height, width)
            
        return d



roi_size = (160, 176, 144)
train_transforms = tf.Compose([
    tf.LoadImaged(keys=["images"],image_only=True, ensure_channel_first=True),
    tf.Orientationd(keys=['images'], axcodes="RAS"),
    tf.EnsureTyped(keys=['images'], dtype=np.float32),
    tf.ToTensord(keys=['images']),
    tf.SpatialPadd(keys=['images'], spatial_size=(160, 176, 144), mode='constant', allow_missing_keys=True),
    tf.CenterSpatialCropd(keys=['images'], roi_size=(160, 176, 144),),
    tf.RandZoomd(keys=['images',], prob=0.2, min_zoom=1.0, max_zoom=1.4, mode=['trilinear'], num_samples=1),
    tf.RandFlipd(keys=['images'], prob=0.2, spatial_axis=[0, 1]),
    tf.RandAdjustContrastd(keys=['images'], prob=0.2, gamma=[1.0, 2.0], retain_stats=True),
    tf.ScaleIntensityRangePercentilesd(keys=['images'], lower=0.5, upper=99.5, b_min=-1, b_max=1, clip=True, relative=False, channel_wise=True),
    RandBlockCombine(keys=["images"], block_size=8),
])

val_transforms = tf.Compose([
    tf.LoadImaged(keys=["images"],image_only=True, ensure_channel_first=True),
    tf.Orientationd(keys=['images'], axcodes="RAS"),
    tf.EnsureTyped(keys=['images'], dtype=np.float32),
    tf.ToTensord(keys=['images']),
    tf.SpatialPadd(keys=['images'], spatial_size=(160, 176, 144), mode='constant', allow_missing_keys=True),
    tf.CenterSpatialCropd(keys=['images'], roi_size=(160, 176, 144),),
    tf.ScaleIntensityRangePercentilesd(keys=['images'], lower=0.5, upper=99.5, b_min=-1, b_max=1, clip=True, relative=False, channel_wise=True),
    RandBlockCombine(keys=["images"], block_size=8),
])
my_model = resnet18(spatial_dims=3, n_input_channels=1, num_classes=2)

# 3. 初始化 Trainer
trainer = Trainer(
    device="cuda:0",
    model=my_model,
    dataset_json="brats2018.json",
    save_dir="experiments/hybrd/",
    batch_size=2,
    max_epochs=100,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    lr=1e-6
)

# 4. 开始训练
trainer.train()