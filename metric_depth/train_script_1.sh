#!/bin/bash

# for scale in $(seq 1.0 -0.1 0.1); do 
#     echo "Running with scale factor: $scale"
#     python3 train_mix_infer.py -d mix --kitti_scale_factor "$scale" 
# done

# python3 train_mix_infer.py -d mix --kitti_scale_factor 0.6 --kitti_ratio 5 --art_ratio 11  --epochs 15

# python3 train_mix_infer.py -d art --epochs 30 --w_mae 1.0 --w_si 0.0 --w_seg_si 0.0 --checkpoint "./depth_anything_finetune/DepthAnythingv1_27-Nov_21-04-28fd8f7d97fe_best.pt"

# for w_seg in $(seq 2.2 0.2 5.0); do
#     echo "Running with w_seg: $w_seg"
#     python3 train_mix_infer.py -d kitti --epochs 10 --w_mae 1.0 --w_si 0.0 --w_seg_si 0.0 --w_seg "$w_seg" --lr 1.6100e-7 --tags "kitti_baseline,w_seg_sweep"
# done

# for w_seg in $(seq 0.6 0.2 1.0); do
#     echo "Running with w_seg: $w_seg"
    
#     # Run your Python script with the current segmentation weight
#     python3 train_mix_infer.py \
#         -d art \
#         --epochs 30 \
#         --w_mae 1.0 \
#         --w_si 0.0 \
#         --w_seg_si 0.0 \
#         --w_seg "$w_seg" \
#         --lr 2.6e-7 \
#         --tags "w_seg_sweep,art_finetune" \
#         --checkpoint "./depth_anything_finetune/DepthAnythingv1_27-Nov_21-04-28fd8f7d97fe_best.pt"
# done    

# # Silog weight sweep
# for w_mae in $(seq 0.05 0.05 0.5); do
#     echo "Running with w_mae: $w_mae"
    
#     # Calculate w_seg_si as (1 - w_mae) using bc for floating-point arithmetic
#     w_si=$(echo "1.0 - $w_mae" | bc)

#     # Run your Python script with the current segmentation weight
#     python3 train_mix_infer.py \
#         -d art \
#         --epochs 20 \
#         --w_mae ${w_mae} \
#         --w_si ${w_si} \
#         --w_seg_si 0.0 \
#         --w_seg 0.0 \
#         --beta 1.0 \
#         --lr 2.8e-7 \
#         --tags "w_mae_silog_sweep,art_finetune" \
#         --checkpoint "./depth_anything_finetune/DepthAnythingv1_27-Nov_21-04-28fd8f7d97fe_best.pt"
# done

# Silog weight sweep

for w_seg in $(seq 0.0 0.1 1.0); do
    echo "Running with w_seg: $w_seg"

    # Run your Python script with the current segmentation weight
    python3 train_mix_infer.py \
        -d art \
        --epochs 20 \
        --w_mae 0.25 \
        --w_si 0.75 \
        --w_seg_si 0.0 \
        --w_seg ${w_seg} \
        --beta 1.0 \
        --lr 2.8e-7 \
        --tags "w_seg_sweep_0.25_w_mae_silog,art_finetune" \
        --checkpoint "./depth_anything_finetune/DepthAnythingv1_27-Nov_21-04-28fd8f7d97fe_best.pt"
done