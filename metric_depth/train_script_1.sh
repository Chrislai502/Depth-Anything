#!/bin/bash

# for scale in $(seq 1.0 -0.1 0.1); do 
#     echo "Running with scale factor: $scale"
#     python3 train_mix_infer.py -d mix --kitti_scale_factor "$scale" 
# done

# python3 train_mix_infer.py -d mix --kitti_scale_factor 0.6 --kitti_ratio 5 --art_ratio 11  --epochs 15

# python3 train_mix_infer.py -d art --epochs 30 --w_mae 1.0 --w_si 0.0 --w_seg_si 0.0 --checkpoint "./depth_anything_finetune/DepthAnythingv1_27-Nov_21-04-28fd8f7d97fe_best.pt"

for w_seg in $(seq 2.2 0.2 5.0); do
    echo "Running with w_seg: $w_seg"
    python3 train_mix_infer.py -d kitti --epochs 10 --w_mae 1.0 --w_si 0.0 --w_seg_si 0.0 --w_seg "$w_seg" --lr 1.6100e-7 --tags "kitti_baseline,w_seg_sweep"
done