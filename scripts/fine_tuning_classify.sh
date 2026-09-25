#!/bin/bash

export LD_LIBRARY_PATH=/home/bmp10/anaconda3/envs/mmm/lib:$LD_LIBRARY_PATH

python train.py \
    --task "cla" \
    --img_size 160 176 144 \
    --patch_size 16 16 16 \
    --in_channels 1 \
    --input_tasks t1n \
    --encoder_embed_dim 768 \
    --encoder_depth 24 \
    --adapter_depth 12 \
    --seg_classes 3 \
    --output_tasks label \
    --checkpoint_path "./logs/pre_train_mask_patch/mdloss_6/20260419_095232/checkpoints/epoch_420_.pth" \
    --keys_remove output_adapters \
    --num_classes 2 \
    --mask_ratio 0.0 \
    --mask_mode "dirichlet" \
    --device "cuda:0" \
    --log_dir "./logs/fine_tuning/cla/mdloss_6/adni" \
    --num_samples 6 \
    --lr 1e-6 \
    --split_json "jsons/adni.json" \
    --batch_size 2 \
    --num_epochs 700 \
    --val_interval 5 \
    --save_interval 5 \
    --classification_task "binary" \
    --multilabel \
    --middle_token