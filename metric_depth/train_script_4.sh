#!/bin/bash

# for scale in $(seq 0.51 0.01 0.69); do 
#     echo "Running with scale factor: $scale"
#     python3 train_mix_infer.py -d mix --kitti_scale_factor "$scale"
# done

# Initial learning rate
lr=7.0e-8

# Number of sweeps (iterations)
num_sweeps=60  # Adjust this to control the number of iterations

for ((i=1; i<=num_sweeps; i++)); do
    echo "Running with lr: $lr"
    
    # Run your Python script with the current learning rate
    python3 train_mix_infer.py \
        -d art \
        --epochs 30 \
        --w_mae 1.0 \
        --w_si 0.0 \
        --w_seg_si 0.0 \
        --w_seg 0.4 \
        --lr "$lr" \
        --tags "lr_sweep,art_finetune" \
        --checkpoint "./depth_anything_finetune/DepthAnythingv1_27-Nov_21-04-28fd8f7d97fe_best.pt"
    
    # Double the learning rate for the next iteration
    lr=$(echo "$lr * 2" | bc -l)
done