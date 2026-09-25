#!/bin/bash

export LD_LIBRARY_PATH=/home/bmp10/anaconda3/envs/mmm/lib:$LD_LIBRARY_PATH

python evaluation.py \
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
    --num_classes 2 \
    --mask_ratio 0.0 \
    --mask_mode "dirichlet" \
    --device "cuda:0" \
    --log_dir "./logs/evaluate/segment/gli" \
    --checkpoint_path "./logs/fine_tuning/seg/cla/20260427_094827/checkpoints/epoch_1_.pth" \
    --num_samples 1 \
    --split_json "jsons/brats2023_gli.json" \
    --batch_size 2 \
    --classification_task "binary" \
    --test_stage \
    --multilabel \