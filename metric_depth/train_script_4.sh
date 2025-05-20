#!/bin/bash

# for scale in $(seq 0.51 0.01 0.69); do 
#     echo "Running with scale factor: $scale"
#     python3 train_mix_infer.py -d mix --kitti_scale_factor "$scale"
# done

# # Initial learning rate
# lr=0.00000028 # 7e-8

# # Number of sweeps (iterations)
# num_sweeps=20  # Adjust this to control the number of iterations

# for ((i=1; i<=num_sweeps; i++)); do
#     echo "Running with lr: $lr"
    
#     # Run your Python script with the current learning rate
#     python3 train_mix_infer.py \
#         -d art \
#         --epochs 30 \
#         --w_mae 1.0 \
#         --w_si 0.0 \
#         --w_seg_si 0.0 \
#         --w_seg 0.4 \
#         --lr "$lr" \
#         --tags "lr_sweep,art_finetune" \
#         --checkpoint "./depth_anything_finetune/DepthAnythingv1_27-Nov_21-04-28fd8f7d97fe_best.pt"
    
#     # Double the learning rate for the next iteration
#     lr=$(echo "$lr * 2" | bc -l)
#     echo "Learning rate: $lr"

# done

# for w_seg in $(seq 1.2 0.2 3.0); do
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
# for w_mae in $(seq 0.1 0.1 1.0); do
#     echo "Running with w_mae: $w_mae"
    
#     # Calculate w_seg_si as (1 - w_mae) using bc for floating-point arithmetic
#     w_si=$(echo "1.0 - $w_mae" | bc)

#     # Run your Python script with the current segmentation weight
#     python3 train_mix_infer.py \
#         -d kitti \
#         --epochs 10 \
#         --w_mae ${w_mae} \
#         --w_si ${w_si} \
#         --w_seg_si 0.0 \
#         --w_seg 0.0 \
#         --beta 1.0 \
#         --lr 1.6100e-7 \
#         --tags "w_mae_silog_sweep,kitti_baseline"
# done

# Silog weight sweep
for w_seg in $(seq 0.0 0.2 2.0); do
    echo "Running with w_seg: $w_seg"

    # Run your Python script with the current segmentation weight
    python3 train_mix_infer.py \
        -d art \
        --epochs 20 \
        --w_mae 1.0 \
        --w_si 0.0 \
        --w_seg_si 0.0 \
        --w_seg ${w_seg} \
        --beta 1.0 \
        --lr 2.8e-7 \
        --tags "w_seg_sweep_1.0_w_mae_silog,art_finetune" \
        --checkpoint "./depth_anything_finetune/DepthAnythingv1_27-Nov_21-04-28fd8f7d97fe_best.pt"
done