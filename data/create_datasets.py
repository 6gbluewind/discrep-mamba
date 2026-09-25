import logging
logger = logging.getLogger(__name__)
import os
import sys
from tqdm import tqdm
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import json
import torch
import pandas as pd
import numpy as np
import data.transforms as transforms


from torch.utils.data import Dataset, DataLoader
from data.data_transforms import get_transforms


class BraTS(Dataset):
    def __init__(self, task, num_samples, is_train=False, split_json=None, img_size=None,modals=None, *args, **kwargs):
        super().__init__()
        self.is_train = is_train
        self.img_size = img_size

        split_json = Path(split_json)
        self.samples = json.loads(split_json.read_text(encoding='utf-8'))
        if self.is_train:
            self.samples = self.samples["train"]
        else:
            self.samples = self.samples["val"]
        
        self.task = task
        self.num_samples = num_samples
        self.transforms = get_transforms(self.task, is_train=is_train, num_samples=self.num_samples, roi_size=img_size,modalities=modals)
      
        self.original_length = len(self.samples)

        print(f'Dataset task: {self.task}, num samples: {self.num_samples}, original samples: {self.original_length}, augment samples: {self.num_samples*self.original_length}')

    def __len__(self):
        if self.num_samples > 1:
            return self.original_length * self.num_samples
        return self.original_length

    def __getitem__(self, index):
        try:
            if self.num_samples > 1:
                original_idx = index // self.num_samples
                sample_idx = index % self.num_samples
            else:
                original_idx = index
                sample_idx = 0

            sample = self.samples[original_idx]
            seed = torch.initial_seed() + index
            torch.manual_seed(seed)

            transformed = self.transforms(sample)

            if isinstance(transformed, list) and self.num_samples > 1:
                item = transformed[sample_idx]
            elif isinstance(transformed, list):
                item = transformed[0]
            else:
                item = transformed

            def convert_to_tensor(data):
                if isinstance(data, torch.Tensor):
                    if hasattr(data, 'as_tensor'):
                        return data.as_tensor().clone()
                    else:
                        return data.clone()
                elif isinstance(data, list):
                    return [convert_to_tensor(sub) for sub in data]
                elif isinstance(data, tuple):
                    return tuple(convert_to_tensor(sub) for sub in data)
                elif isinstance(data, dict):
                    return {key: convert_to_tensor(val) for key, val in data.items()}
                else:
                    return data

            return convert_to_tensor(item)

        except Exception as e:
            raise e
        
    

def get_loader(task, num_samples, is_train, split_json, img_size, num_workers=4, batch_size=2, pin_memory=True, shuffle=None,modals=None):
    if shuffle is None:
        shuffle = is_train
    dataset = BraTS(task, num_samples, is_train, split_json, img_size,modals=modals)
    
    return DataLoader(
        dataset=dataset, 
        batch_size=batch_size,
        pin_memory=pin_memory, 
        num_workers=num_workers, 
        shuffle=shuffle,
        drop_last=True
    )


if __name__ == '__main__':
    num_samples = 8
    is_train = True
    dataloader = get_loader(None, 'cla', num_samples, is_train, './jsons/test.json', img_size=[160,176,144], tasks=['t1n', 't1c', 't2w', 't2f'], num_workers=0)
    print(len(dataloader))
    t1_list = []
    for data in dataloader:
       t1_list.append(data['t1c'])
    
    print(torch.equal(t1_list[0], t1_list[1]))
    print(torch.equal(t1_list[1], t1_list[2]))
    print(torch.equal(t1_list[2], t1_list[3]))