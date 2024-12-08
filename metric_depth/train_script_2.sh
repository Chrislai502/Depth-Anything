#!/bin/bash

# for scale in $(seq 0.5 0.05 0.7); do 
#     echo "Running with scale factor: $scale"
#     python3 train_mix_infer.py -d mix --kitti_scale_factor "$scale" --kitti_ratio 5 --art_ratio 11 --epoch 10
# done

python3 train_mix_infer.py -d kitti --epochs 13 --w_mae 1.0 --w_si 0.0 --w_seg_si 0.0 --w_seg 0.2 --lr 1.6100e-7