import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from utils import get_transform, create_logger, tensor2array, get_samples
from monai import transforms as tfs

from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure

logging_dir = './output_analysis'
os.makedirs(logging_dir, exist_ok=True)

logger = create_logger(logging_dir=logging_dir)

def compare_modalities(sample, transforms=None):
    if transforms:
        sample = transforms(sample)
    ssim = StructuralSimilarityIndexMeasure(data_range=(-1.0, 1.0))
    # sample = tensor2array(sample)
    metrics = {}
    
    # SSIM
    metrics['SSIM_T1'] = ssim(sample['t1'], sample['t1ce'])
    metrics['SSIM_T2'] = ssim(sample['t2'], sample['t1ce'])
    metrics['SSIM_FLAIR'] = ssim(sample['flair'], sample['t1ce'])
    
    # Pearsonr
    metrics['Pearson_T1'] = pearsonr(sample['t1'].flatten(), sample['t1ce'].flatten())[0]
    metrics['Pearson_T2'] = pearsonr(sample['t2'].flatten(), sample['t1ce'].flatten())[0]
    metrics['Pearson_FLAIR'] = pearsonr(sample['flair'].flatten(), sample['t1ce'].flatten())[0]
    
    # MSE
    metrics['MSE_T1'] = np.mean((sample['t1'] - sample['t1ce'])**2)
    metrics['MSE_T2'] = np.mean((sample['t2'] - sample['t1ce'])**2)
    metrics['MSE_FLAIR'] = np.mean((sample['flair'] - sample['t1ce'])**2)
    
    return metrics

def data_analysis(data_root, keys=['t1', 't1ce', 't2', 'flair'], roi_size=(160, 176, 144)):
    samples, sample_name = get_samples(data_root)
    transforms = tfs.Compose(get_transform(keys=keys, roi_size=roi_size))
    metrics = {}
    for sample in samples:
        mes = compare_modalities(sample, transforms)
        for k, v in mes.items():
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(v)
    avg_metrics = {
        k: sum(v)/len(v)
        for k, v in metrics.items()
    }

    for k, v in avg_metrics.items():
        logger.info(f'{k} {v:.4f}')

if __name__ == '__main__':
    # sample = {
    #     't1': '/Users/wang/private/datasets/BraTS2023-GLI/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1n.nii.gz',
    #     't1ce': '/Users/wang/private/datasets/BraTS2023-GLI/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1c.nii.gz',
    #     't2': '/Users/wang/private/datasets/BraTS2023-GLI/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2w.nii.gz',
    #     'flair': '/Users/wang/private/datasets/BraTS2023-GLI/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2f.nii.gz'
    # }
    # transforms = tfs.Compose(get_transform(sample.keys(), roi_size=(160,176,144)))
    # metrics = compare_modalities(sample, transforms)
    # for k, v in metrics.items():
    #     logger.info(f'{k} {v:.4f}')
    data_root = '/media/bmp1/store5/dataset/brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
    roi_size = (160, 176, 144)
    keys=['t1', 't1ce', 't2', 'flair']
    data_analysis(data_root, keys, roi_size)