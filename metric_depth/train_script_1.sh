#!/bin/bash

# for scale in $(seq 1.0 -0.1 0.1); do 
#     echo "Running with scale factor: $scale"
#     python3 train_mix_infer.py -d mix --kitti_scale_factor "$scale" 
# done

# python3 train_mix_infer.py -d mix --kitti_scale_factor 0.6 --kitti_ratio 5 --art_ratio 11  --epochs 15

python3 train_mix_infer.py -d art --epochs 30 --w_mae 1.0 --w_si 0.0 --w_seg_si 0.0 --checkpoint "./depth_anything_finetune/DepthAnythingv1_27-Nov_21-04-28fd8f7d97fe_best.pt"