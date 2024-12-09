#!/bin/bash

# for scale in $(seq 0.5 0.05 0.7); do 
#     echo "Running with scale factor: $scale"
#     python3 train_mix_infer.py -d mix --kitti_scale_factor "$scale" --kitti_ratio 5 --art_ratio 11 --epoch 10
# done

# python3 train_mix_infer.py -d kitti --epochs 13 --w_mae 1.0 --w_si 0.0 --w_seg_si 0.0 --w_seg 0.2 --lr 1.6100e-7

# Initial learning rate
lr=0.005 # 5e-3

# Number of sweeps (iterations)
num_sweeps=20  # Adjust this to control the number of iterations

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
    lr=$(echo "$lr / 2" | bc -l)
    echo "Learning rate: $lr"
done