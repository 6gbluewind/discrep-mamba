#!/bin/bash

export LD_LIBRARY_PATH=/home/bmp10/anaconda3/envs/mmm/lib:$LD_LIBRARY_PATH

python train.py \
    --task "seg" \
    --img_size 128 128 128 \
    --patch_size 16 16 16 \
    --in_channels 1 \
    --input_tasks t1n t1c t2w t2f \
    --encoder_embed_dim 768 \
    --encoder_depth 24 \
    --adapter_depth 6 \
    --seg_classes 3 \
    --output_tasks seg \
    --checkpoint_path "logs/pre_train_mask_patch/mdloss_6/20260419_095232/checkpoints/epoch_420_.pth" \
    --keys_remove embed output_adapters \
    --num_classes 2 \
    --mask_ratio 0.0 \
    --mask_mode "dirichlet" \
    --device "cuda:0" \
    --log_dir "./logs/fine_tuning/seg/gli" \
    --num_samples 1 \
    --lr 1e-4 \
    --split_json "jsons/brats2023_gli.json" \
    --batch_size 2 \
    --num_epochs 700 \
    --val_interval 1 \
    --save_interval 1 \
    --classification_task "binary" \
    --multilabel \
