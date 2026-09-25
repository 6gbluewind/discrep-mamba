import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from monai.networks.nets import resnet18, resnet50
import monai.transforms as tf
import numpy as np
from resnet_trainer import Trainer



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
])

val_transforms = tf.Compose([
    tf.LoadImaged(keys=["images"],image_only=True, ensure_channel_first=True),
    tf.Orientationd(keys=['images'], axcodes="RAS"),
    tf.EnsureTyped(keys=['images'], dtype=np.float32),
    tf.ToTensord(keys=['images']),
    tf.SpatialPadd(keys=['images'], spatial_size=(160, 176, 144), mode='constant', allow_missing_keys=True),
    tf.CenterSpatialCropd(keys=['images'], roi_size=(160, 176, 144),),
    tf.ScaleIntensityRangePercentilesd(keys=['images'], lower=0.5, upper=99.5, b_min=-1, b_max=1, clip=True, relative=False, channel_wise=True),
])
my_model = resnet18(spatial_dims=3, n_input_channels=1, num_classes=2)

# 3. 初始化 Trainer
trainer = Trainer(
    device="cuda:0",
    model=my_model,
    dataset_json="brats2018_t1.json",
    save_dir="experiments/4resnet_t1/",
    batch_size=2,
    max_epochs=100,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    lr=1e-6
)

# 4. 开始训练
trainer.train()