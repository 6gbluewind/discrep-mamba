import torch
from torch import Tensor
import numpy as np
import monai.transforms as tf
from monai.transforms import MapTransform
from monai.config import KeysCollection
from typing import Hashable, Mapping
from monai.data import MetaTensor
from copy import deepcopy
from torch import Tensor


def get_transforms(task, modalities=['t1n', 't1c', 't2w', 't2f'], is_train=True, roi_size=None, num_samples=1):
    
    if task == 'rec':
        transforms = reconstruct(num_samples, roi_size)
    elif task == 'cla':
        transforms = classification(num_samples, roi_size)
    elif task == 'seg':
        transforms = segmentation(num_samples, roi_size)
    else:
        raise ValueError(f"don't defind: {task}")
    
    shape_transforms, norm_transforms, augment_transforms, seg_transforms = transforms
    
    load_transforms = [
        tf.LoadImaged(keys=["images", "seg"],image_only=False,ensure_channel_first=True,allow_missing_keys=True,),
        tf.Orientationd(keys=['images', 'seg'], axcodes="RAS", allow_missing_keys=True),
        tf.EnsureTyped(keys=['images', 'seg'], dtype=np.float32, allow_missing_keys=True),
        tf.ToTensord(keys=['imaegs', 'seg'], allow_missing_keys=True),
    ]

    cleanup_transforms = [
        tf.SplitDimd(keys="images", output_postfixes=modalities, dim=0, update_meta=True),
        tf.DeleteItemsd(keys=["images"]),
        tf.Lambda(func=lambda data: {k.replace("images_", ""): v for k, v in data.items()}),
    ]
    
    all_transforms = []
    # 1. base load transforms
    all_transforms.extend(load_transforms)

    # 2. shape transforms
    all_transforms.extend(shape_transforms)
    
    # 3. data augment transforms
    if is_train:
        all_transforms.extend(augment_transforms)
    
    # 4. norm transforms
    all_transforms.extend(norm_transforms)
    
    # 5. seg transforms
    if seg_transforms:
        all_transforms.extend(seg_transforms)

    # 6. cleanup transforms
    all_transforms.extend(cleanup_transforms)
    
    return tf.Compose(all_transforms)


def reconstruct(num_samples=2, roi_size=(160, 176, 144)):
    """reconstruct transforms"""
    shape_transforms = [
        tf.SpatialPadd(keys=['images', 'seg'], spatial_size=roi_size, mode='constant', allow_missing_keys=True),
        tf.CenterSpatialCropd(
            keys=['images', 'seg'], 
            allow_missing_keys=True, 
            roi_size=roi_size,
        ),
       
    ]
    
    augment_transforms = [
        tf.RandZoomd(
            keys=['images','seg'], 
            allow_missing_keys=True, 
            prob=0.2, 
            min_zoom=1.0, 
            max_zoom=1.4,
            mode=['trilinear', 'nearest-exact'],
            num_samples=num_samples,
        ),
        tf.RandFlipd(
            keys=['images','seg'], 
            allow_missing_keys=True, 
            prob=0.2, 
            spatial_axis=[0, 1],
            
        ),
        tf.RandAdjustContrastd(
            keys=['images'], 
            allow_missing_keys=True, 
            prob=0.2, 
            gamma=[1.0, 2.0], 
            retain_stats=True,
        )
    ]
    
    norm_transforms = [
        tf.ScaleIntensityRangePercentilesd(
            keys=['images'], 
            lower=0.5, 
            upper=99.5, 
            b_min=-1, 
            b_max=1, 
            clip=True, 
            relative=False, 
            channel_wise=True
        ),
    ]
    
    return shape_transforms, norm_transforms, augment_transforms, []


def classification(num_samples=2, roi_size=(160, 176, 144)):
    """classification transforms"""
    shape_transforms = [
        tf.SpatialPadd(keys=['images', 'seg'], spatial_size=roi_size, mode='constant', allow_missing_keys=True),
        tf.CenterSpatialCropd(
            keys=['images', 'seg'], 
            allow_missing_keys=True, 
            roi_size=roi_size
        ),
       tf.RandSpatialCropSamplesd(
            keys=['images'],
            roi_size=roi_size,
            num_samples=num_samples,
            random_size=False,
        )
    ]
    augment_transforms = [
        tf.RandZoomd(
            keys=['images','seg'], 
            allow_missing_keys=True, 
            prob=0.2, 
            min_zoom=1.0, 
            max_zoom=1.4,
            mode=['trilinear', 'nearest-exact']
        ),
        tf.RandFlipd(
            keys=['images','seg'], 
            allow_missing_keys=True, 
            prob=0.2, 
            spatial_axis=[0, 1]
        ),
        tf.RandAffined(
            keys=['images','seg'],
            allow_missing_keys=True, 
            prob=0.2, 
            mode=['trilinear','nearest'], 
            rotate_range=0.349, 
            scale_range=0.1
        ),
        tf.RandGaussianSmoothd(
            keys=['images'], 
            allow_missing_keys=True, 
            prob=0.2, 
            sigma_x=(0.0, 0.1), 
            sigma_y=(0.0, 1.5), 
            sigma_z=(0.0, 1.5)
        ),
        tf.RandGaussianNoised(
            keys=['images'], 
            allow_missing_keys=True, 
            prob=0.2, 
            std=0.1 
        ),
        tf.RandAdjustContrastd(
            keys=['images', 'seg'], 
            allow_missing_keys=True, 
            prob=0.2, 
            gamma=[1.0, 2.0], 
            retain_stats=True
        ),
        
    ]
    
    norm_transforms = [
        tf.ScaleIntensityRangePercentilesd(
            keys=['images'], 
            lower=0.5, 
            upper=99.5, 
            b_min=-1, 
            b_max=1, 
            clip=True, 
            relative=False, 
            channel_wise=True
        ),
    ]
    
    return shape_transforms, norm_transforms, augment_transforms, []


def segmentation(num_samples=6 ,target_size=(240, 240, 160), roi_size=(128, 128, 128)):
    """segment transform"""
    
    shape_transforms = [
        tf.SpatialPadd(keys=['images', 'seg'], allow_missing_keys=True, spatial_size=target_size, mode='constant'),
        # AddCropPositiond(keys=['images','seg'], original_size=(240,240,155), target_size=(240,240,160), allow_missing_keys=True),
        tf.RandCropByPosNegLabeld(
            keys=['images', 'seg'],
            label_key='seg',
            image_key='images',
            spatial_size=roi_size,
            pos=1,
            neg=1,
            num_samples=num_samples,  # add samples
            image_threshold=0,
        ),
    ]
    
    augment_transforms = [
        tf.RandZoomd(
            keys=['images','seg'], 
            allow_missing_keys=True, 
            prob=0.5,
            min_zoom=0.9, 
            max_zoom=1.1, 
            mode=['trilinear', 'nearest-exact']
        ),
        tf.RandFlipd(
            keys=['images','seg'], 
            allow_missing_keys=True, 
            prob=0.5, 
            spatial_axis=[0, 1, 2] 
        ),
        tf.RandAdjustContrastd(
            keys=['images'], 
            allow_missing_keys=True, 
            prob=0.3, 
            gamma=(0.7, 1.3), 
            retain_stats=True
        )
    ]
    
    norm_transforms = [
        tf.ScaleIntensityRangePercentilesd(
            keys=['images'], 
            lower=0.5, 
            upper=99.5, 
            b_min=-1, 
            b_max=1, 
            clip=True, 
            relative=False, 
            channel_wise=True
        ),

    ]
    
    seg_transforms = [
        ConvertBraTSClassesToMultiLabelsd(keys=['seg'])
    ]
    
    return shape_transforms, norm_transforms, augment_transforms, seg_transforms


class ConvertBraTSClassesToMultiLabelsd(MapTransform):
    """
    Convert BraTS labels to multi-channel format.
    
    For BraTS 2023 (labels: 0,1,2,3):
    - Label 1: Non-enhancing tumor core (NCR/NET)
    - Label 2: Peritumoral edema (ED)
    - Label 3: GD-enhancing tumor (ET)

    For BraTS 2018 (labels: 0,1,2,4):
    - Label 1: Non-enhancing tumor core (NCR/NET)
    - Label 2: Peritumoral edema (ED)
    - Label 3: GD-enhancing tumor (ET)
    """
    
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        brats_version: str = "2023"
    ):
        super().__init__(keys, allow_missing_keys)
        self.brats_version = brats_version
        
        if brats_version == "2023":
            self.et_label = 3
        elif brats_version == "2018":
            self.et_label = 4
        else:
            raise ValueError(f"Unsupported BRATS version: {brats_version}")
    
    def __call__(self, data):
        d = dict(data)
        
        for key in self.key_iterator(d):
            label = d[key]
            if not torch.is_tensor(label):
                label = torch.as_tensor(label, dtype=torch.int16)
            
            # create channels
            wt = ((label == 1) | (label == 2) | (label == self.et_label)).float()
            tc = ((label == 1) | (label == self.et_label)).float()
            et = (label == self.et_label).float()
        
            # cat: [ET, TC, WT]
            result = torch.cat([et, tc, wt], dim=0)

            if isinstance(label, MetaTensor):
                result = MetaTensor(result, meta=label.meta)
            
            d[key] = result

        return d

class MultiLabelsToBraTSClasses(tf.Transform):    
    def __init__(self, label_offset = 0):
        self.label_offset = label_offset
    
    def __call__(self, outputs: Tensor):    
        et_channel = outputs[:, self.label_offset+0:self.label_offset+1, ...]  # channel 0: et
        tc_channel = outputs[:, self.label_offset+1:self.label_offset+2, ...]  # channel 1: tc
        wt_channel = outputs[:, self.label_offset+2:self.label_offset+3, ...]  # channel 2: wt
        
        # init single channel label 0
        single_channel = torch.zeros_like(wt_channel, dtype=torch.long)
      
        # 1. ET: (3)
        # ET = (ET channel == 1)
        single_channel[et_channel == 1] = self.label_offset + 3
        
        # 2. NCR/NET (1)
        # NCR/NET = TC channel == 1 and  ET channel != 1
        ncr_net_mask = (tc_channel == 1) & (et_channel != 1)
        single_channel[ncr_net_mask] = self.label_offset + 1
        
        # 3. ED: (2)
        # ED = ( WT channel == 1 and  TC channel != 1)
        ed_mask = (wt_channel == 1) & (tc_channel != 1)
        single_channel[ed_mask] = self.label_offset + 2
        
        return single_channel
  