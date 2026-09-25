import torch
import logging
import nibabel as nib
import numpy as np
import monai.transforms as tf
from pathlib import Path 
from tqdm import tqdm

def tensor2array(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, dict):
        res = {}
        for k, v in data.items():
            res[k] = tensor2array(v)
        return res
    elif isinstance(data, np.array):
        return data

def data_analysis(sample):
    """calculate maximum, minimum, mean, std"""
    if isinstance(sample, dict):
        maximum, minimum, mean, std = {}, {}, {}, {}
        for k, v in sample.items():
            maximum[k] = np.amax(v)
            minimum[k] = np.amin(v)
            mean[k] = np.mean(v)
            std[k] = np.std(v)

        return  maximum, minimum, mean, std

    return sample.max(), sample.min(), sample.mean(), sample.std()


def load_nifit(data, to_array=True):
    """load nifit data"""

    if isinstance(data, dict):
        fdata = {}
        for k, v in data.items():
            fdata[k] = load_nifit(v, to_array=to_array)
        return fdata

    img = nib.load(data)
    fdata = img.get_fdata()
    if to_array:
        fdata = np.array(fdata)

    return fdata

def create_logger(logging_dir='.'):
    """Create a logger that writes to a log file and stdout."""

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)

    return logger


def get_transform(keys, roi_size=(160, 176, 144)):
    """get pre process transform"""

    return [
        tf.LoadImaged(keys=keys, image_only=True, ensure_channel_first=True, allow_missing_keys=True,),
        tf.Orientationd(keys=keys, axcodes="RAS", allow_missing_keys=True, labels=None),
        tf.EnsureTyped(keys=keys, dtype=np.float32, allow_missing_keys=True),
        tf.ToTensord(keys=keys, allow_missing_keys=True),
        tf.CenterSpatialCropd(keys=keys,  allow_missing_keys=True, roi_size=roi_size),
        tf.ScaleIntensityRangePercentilesd(keys=keys, lower=0.5, upper=99.5, b_min=-1, b_max=1, clip=True, relative=False, channel_wise=True),
    ]


def load_and_transform(sample, transforms, flatten=False):
    """load and aply transforms to image"""

    transformed_sample = {}
    # aply transforms
    transformed = transforms(sample)
    for k, v in transformed.items():
        if hasattr(v, 'numpy'):  # tensor
            if flatten:
                transformed_sample[k] = v.numpy().flatten()
            else:
                transformed_sample[k] = v.numpy()
        else:  # numpy
            if flatten:
                transformed_sample[k] = v.flatten()
            else:
                transformed_sample[k] = v

    return transformed_sample

def get_samples(data_root):
    """from data root collect data"""
    
    samples = []
    sample_names = []
    data_root = Path(data_root)

    sample_folders = list(data_root.glob("*"))
    for p in tqdm(sample_folders, desc="Collecting samples"):
        sample = {}
        for m in p.glob("*t1n*"):
            sample['t1'] = str(m)
        for m in p.glob("*t1c*"):
            sample['t1ce'] = str(m)
        for m in p.glob("*t2w*"):
            sample['t2'] = str(m)
        for m in p.glob("*t2f*"):
            sample['flair'] = str(m)

        if len(sample.keys()) == 4:
            samples.append(sample)
            sample_names.append(p.name)

    return samples, sample_names


def standardization(
    data: np.ndarray, 
    axis: bool = None,
    ddof: int =0
):
    if isinstance(data, dict):
        res = {}
        for k, v in data.items():
            res[k] = standardization(v, axis=axis, ddof=ddof)
        return res

    if isinstance(data, np.ndarray):
        data = np.asarray(data, dtype=np.float32)
    
    # calculate mean and std
    mean_val = np.mean(data, axis=axis, keepdims=True if axis is not None else False)
    std_val = np.std(data, axis=axis, ddof=ddof, keepdims=True if axis is not None else False)
    
    # avoid divide zero
    std_val = np.where(std_val == 0, 1e-8, std_val)
    
    return (data - mean_val) / std_val


def normaliation(
    data: np.ndarray, 
    data_range: tuple[int, int] = (0, 1)
):
    if isinstance(data, dict):
        res = {}
        for k, v in data.items():
            res[k] = normaliation(v, data_range=data_range)
        return res
    
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    data = data.astype(np.float32)
    min = data.min()
    max = data.max()

    if min == max:
        return np.zeros_like(data) + data_range[0]
    
    normalized = (data - min) / (max - min)
    if data_range != (0, 1):
        a, b = data_range
        normalized = normalized * (b - a) + a
    
    return normalized

def squeeze(data):
    if isinstance(data, torch.Tensor):
        return data.squeeze(0)
    elif isinstance(data, np.ndarray):
        return data[0]
    elif isinstance(data, dict):
        res = {}
        for k, v in data.items():
            res[k] = squeeze(v)
        return res