import os
import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
from utils import get_transform, create_logger, load_and_transform, data_analysis


import matplotlib
import monai.transforms as tf

# set matplotlib
matplotlib.use('Agg')

log_root = './outputs_trans'
log_img = './outputs_trans/images'
log_sta = './outputs_trans/statistics'
# create outputs dir
os.makedirs(log_root, exist_ok=True)
os.makedirs(log_img, exist_ok=True)
os.makedirs(log_sta, exist_ok=True)

MODALITYS = ['t1', 't1ce', 't2', 'flair']
COLORS = ['blue', 'green', 'red', 'orange']

logger = create_logger(log_root)
    



def histogram(sample_arr, sample_name, is_transformed=False):
    """Draw and save grayscale histogram for a single sample"""
    try:
        plt.figure(figsize=(16, 10))
        
        for idx, modality in enumerate(MODALITYS):
            plt.subplot(2, 2, idx + 1)
            
            data = sample_arr[modality]
            
            # reomve background pixel
            data_filtered = data[data > np.percentile(data, 1)]
        
            # calculate histogram
            hist, bins = np.histogram(data_filtered, bins=200, density=True)
            
            # draw histogram
            plt.bar(bins[:-1], hist, width=(bins[1] - bins[0]) * 0.8, alpha=0.7, color=COLORS[idx], edgecolor='black')
            
            # add statistics
            stats_text = f"Max: {np.max(data):.1f}\nMin: {np.min(data):.1f}\nMean: {np.mean(data):.2f}\nStd: {np.std(data):.2f}"
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            title = f'{modality.upper()} Histogram'
            if is_transformed:
                title += ' (Transformed)'
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel('Intensity Value', fontsize=12)
            plt.ylabel('Probability Density', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            
        title = f'Grayscale Histograms - Sample: {sample_name}'
        if is_transformed:
            title += ' (After Transforms)'
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # save figure
        suffix = "transformed" if is_transformed else "raw"
        save_path = f"{log_img}/histogram_{sample_name}_{suffix}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Histogram saved to: {save_path}")
        
    except Exception as e:
        logger.error(f"Error creating histogram for sample {sample_name}: {str(e)}")
        plt.close('all')

def save_statistics_to_csv(all_max, all_min, all_mean, all_std, sample_names, suffix=""):
    """Save statistical data to CSV file"""
    try:
        import pandas as pd
        
        stats_data = []
        for idx, sample_name in enumerate(sample_names):
            for modality in MODALITYS:
                stats_data.append({
                    'Sample': sample_name,
                    'Modality': modality,
                    'Max': all_max[modality][idx] if idx < len(all_max[modality]) else np.nan,
                    'Min': all_min[modality][idx] if idx < len(all_min[modality]) else np.nan,
                    'Mean': all_mean[modality][idx] if idx < len(all_mean[modality]) else np.nan,
                    'Std': all_std[modality][idx] if idx < len(all_std[modality]) else np.nan
                })
        
        filename = "brain_mri_statistics"
        if suffix:
            filename += f"_{suffix}"
        csv_path = f"{log_sta}/{filename}.csv"
        df = pd.DataFrame(stats_data)
        df.to_csv(csv_path, index=False)
        logger.info(f"Statistics saved to: {csv_path}")
        
        # save statistics
        summary_data = []
        for modality in MODALITYS:
            if all_max[modality]: 
                summary_data.append({
                    'Modality': modality,
                    'Avg_Max': np.mean(all_max[modality]),
                    'Avg_Min': np.mean(all_min[modality]),
                    'Avg_Mean': np.mean(all_mean[modality]),
                    'Avg_Std': np.mean(all_std[modality])
                })
        
        summary_filename = "summary_statistics"
        if suffix:
            summary_filename += f"_{suffix}"
        df_summary = pd.DataFrame(summary_data)
        summary_path = f"{log_sta}/{summary_filename}.csv"
        df_summary.to_csv(summary_path, index=False)
        logger.info(f"Summary statistics saved to: {summary_path}")
        
    except ImportError:
        logger.warning("Pandas not installed. Skipping CSV export.")
    except Exception as e:
        logger.error(f"Error saving statistics to CSV: {str(e)}")

def show_grayscale_histogram(data_path, analyze_transformed=True):
    samples = []
    sample_names = []
    data_path = Path(data_path)
    
    sample_folders = list(data_path.glob("*"))
    logger.info(f"Found {len(sample_folders)} sample folders")
    
    # get transforms
    transforms = tf.Compose(get_transform(MODALITYS, roi_size=(240, 240, 255)))


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

    logger.info(f"Processing {len(samples)} valid samples")
    

    raw_stats = {
        'max': {k:[] for k in MODALITYS}, 
        'min': {k:[] for k in MODALITYS}, 
        'mean': {k:[] for k in MODALITYS}, 
        'std': {k:[] for k in MODALITYS}
    }
    
    transformed_stats = {
        'max': {k:[] for k in MODALITYS}, 
        'min': {k:[] for k in MODALITYS}, 
        'mean': {k:[] for k in MODALITYS}, 
        'std': {k:[] for k in MODALITYS}
    }
    
    for idx, (sample, sample_name) in enumerate(zip(samples, sample_names)):
        logger.info(f"\nProcessing sample {idx+1}/{len(samples)}: {sample_name}")
        
        # 1. raw data
        raw_sample = {}
        for k, v in sample.items():
            img = nib.load(v)
            raw_sample[k] = img.get_fdata().flatten()
        
        raw_maximum, raw_minimum, raw_mean, raw_std = data_analysis(raw_sample)
        
        for k in MODALITYS:
            raw_stats['max'][k].append(raw_maximum[k])
            raw_stats['min'][k].append(raw_minimum[k])
            raw_stats['mean'][k].append(raw_mean[k])
            raw_stats['std'][k].append(raw_std[k])
        
        histogram(raw_sample, sample_name, is_transformed=False)
        
        # 2. transformed data
        transformed_sample = load_and_transform(sample, transforms, True)
        
        trans_maximum, trans_minimum, trans_mean, trans_std = data_analysis(transformed_sample)
        
        for k in MODALITYS:
            transformed_stats['max'][k].append(trans_maximum[k])
            transformed_stats['min'][k].append(trans_minimum[k])
            transformed_stats['mean'][k].append(trans_mean[k])
            transformed_stats['std'][k].append(trans_std[k])
    
        histogram(transformed_sample, sample_name, is_transformed=True)             
        
        # 记录统计信息
        logger.info(f"{'='*80}")
        logger.info(f"Raw statistics for {sample_name}:")
        for k in MODALITYS:
            logger.info(f"{k.upper():<6}  max: {raw_maximum[k]:^12.4f}  min: {raw_minimum[k]:^12.4f}  mean: {raw_mean[k]:^12.4f}  std: {raw_std[k]:^12.4f}")
        logger.info(f"{'-'*80}")
        logger.info(f"Transformed statistics for {sample_name}:")
        for k in MODALITYS:
            logger.info(f"{k.upper():<6}  max: {trans_maximum[k]:^12.4f}  min: {trans_minimum[k]:^12.4f}  mean: {trans_mean[k]:^12.4f}  std: {trans_std[k]:^12.4f}")

        del raw_sample, raw_maximum, raw_minimum, raw_mean, raw_std
        del transformed_sample, trans_maximum, trans_minimum, trans_mean, trans_std
            
    
    save_statistics_to_csv(raw_stats['max'], raw_stats['min'], raw_stats['mean'], raw_stats['std'], sample_names, suffix="raw")
    
    save_statistics_to_csv(transformed_stats['max'], transformed_stats['min'], 
                           transformed_stats['mean'], transformed_stats['std'], sample_names, suffix="transformed")
        
    

    logger.info("raw data overall statistics")
    for k in MODALITYS:
        avg_max = np.mean(raw_stats['max'][k])
        avg_min = np.mean(raw_stats['min'][k])
        avg_mean = np.mean(raw_stats['mean'][k])
        avg_std = np.mean(raw_stats['std'][k])
        
        logger.info(f"{k.upper():<6} avg_max: {avg_max:>10.4f}, avg_min: {avg_min:>10.4f}, avg_mean: {avg_mean:>10.4f}, avg_std: {avg_std:>10.4f}")
    
    logger.info("transformed data overall statistics")
    for k in MODALITYS:
        avg_max = np.mean(transformed_stats['max'][k])
        avg_min = np.mean(transformed_stats['min'][k])
        avg_mean = np.mean(transformed_stats['mean'][k])
        avg_std = np.mean(transformed_stats['std'][k])
        
        logger.info(f"{k.upper():<6} avg_max: {avg_max:>10.4f}, avg_min: {avg_min:>10.4f}, avg_mean: {avg_mean:>10.4f}, avg_std: {avg_std:>10.4f}")
    
    # create overall histograms
    create_overall_histograms(raw_stats['mean'], sample_names, "raw")
    create_overall_histograms(transformed_stats['mean'], sample_names, "transformed")

def create_overall_histograms(all_mean, sample_names, suffix=""):
    """Create histograms showing distribution of mean values across all samples"""
    try:
        plt.figure(figsize=(14, 10))
        
        for idx, modality in enumerate(MODALITYS):
            plt.subplot(2, 2, idx + 1)
            
            data = all_mean[modality]
            
            # draw mean histogram
            n, bins, patches = plt.hist(data, bins=20, alpha=0.7, 
                                       color=COLORS[idx], edgecolor='black')
            
            plt.axvline(x=np.mean(data), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(data):.2f}')
            plt.axvline(x=np.median(data), color='green', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(data):.2f}')
            
            title = f'{modality.upper()} - Mean Value Distribution'
            if suffix:
                title += f' ({suffix})'
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel('Mean Intensity Value', fontsize=12)
            plt.ylabel('Number of Samples', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            stats_text = f"Samples: {len(data)}\nStd: {np.std(data):.2f}\nRange: [{np.min(data):.2f}, {np.max(data):.2f}]"
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        title = 'Distribution of Mean Intensity Values Across All Samples'
        if suffix:
            title += f' ({suffix})'
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = "overall_statistics"
        if suffix:
            filename += f"_{suffix}"
        save_path = f"{log_img}/{filename}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Overall statistics histogram saved to: {save_path}")
        
    except Exception as e:
        logger.error(f"Error creating overall histograms: {str(e)}")
        plt.close('all')

if __name__ == "__main__":
    data_path = r"/media/bmp1/store5/dataset/brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    
    show_grayscale_histogram(data_path, analyze_transformed=True)
        