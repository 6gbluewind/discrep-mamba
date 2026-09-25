#!/bin/bash

export LD_LIBRARY_PATH=/home/bmp10/anaconda3/envs/mmm/lib:$LD_LIBRARY_PATH

python evaluation.py \
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
    --num_classes 2 \
    --mask_ratio 0.0 \
    --mask_mode "dirichlet" \
    --device "cuda:0" \
    --log_dir "./logs/evaluate_classify/brainmvp" \
    --checkpoint_path "./logs/fine_tuning/cla/brainmvp_adni/20260413_100727/checkpoints/epoch_30_.pth" \
    --num_samples 1 \
    --split_json "jsons/brainmvp_adni.json" \
    --batch_size 2 \
    --classification_task "binary" \
    --test_stage \
    --multilabel \
    --middle_token