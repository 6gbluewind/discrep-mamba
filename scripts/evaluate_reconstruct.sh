#!/bin/bash

export LD_LIBRARY_PATH=/home/bmp10/anaconda3/envs/mmm/lib:$LD_LIBRARY_PATH

python evaluation.py \
    --task "rec" \
    --img_size 160 176 144 \
    --patch_size 16 16 16 \
    --in_channels 1 \
    --input_tasks t1n t1c t2w t2f \
    --encoder_embed_dim 768 \
    --encoder_depth 24 \
    --adapter_depth 6 \
    --seg_classes 3 \
    --output_tasks t1n t1c t2w t2f \
    --num_classes 2 \
    --mask_ratio 0.75 \
    --mask_mode "dirichlet" \
    --device "cuda:0" \
    --log_dir "./logs/evaluate_reconstruct/paper_reconstruct" \
    --checkpoint_path "logs/pre_train_mask_patch/mdloss_6/20260419_095232/checkpoints/epoch_420_.pth" \
    --num_samples 1 \
    --split_json "./jsons/image.json" \
    --batch_size 2 \
    --classification_task "binary" \
    --test_stage \
    --multilabel \
    --middle_token