#!/bin/bash

# for scale in $(seq 0.0 1.0 16.0); do 
#     echo "Running with Kitti Ratio: $scale" # art_ratio will be 16 - $scale
#     python3 train_mix_infer.py -d mix --kitti_scale_factor 0.6 --kitti_ratio "$scale" --art_ratio "$(echo "16 - $scale" | bc)" --epoch
# done

# for scale in $(seq 0.0 1.0 16.0); do 
#     echo "Running with Kitti Ratio: $scale" # art_ratio will be 16 - $scale
#     python3 train_mix_infer.py -d mix --kitti_scale_factor 0.6 --kitti_ratio "$scale" --art_ratio "$(echo "16 - $scale" | bc)" --epoch
# done

# for scale in $(seq 0.45 0.05 0.75); do 
#     echo "Running with scale factor: $scale"
#     python3 train_mix_infer.py -d mix --kitti_scale_factor "$scale" --kitti_ratio 8 --art_ratio 8 --epoch 10 --merge_batches 0
# done

# for scale in $(seq 0.45 0.05 0.75); do 
#     echo "Running with scale factor: $scale"
#     python3 train_mix_infer.py -d kitti --kitti_scale_factor "$scale" --epoch 5
# done

for w_seg in $(seq 0.0 0.1 1.0); do
    echo "Running with w_seg: $w_seg"
    python3 train_mix_infer.py -d kitti --epochs 9 --w_mae 1.0 --w_si 0.0 --w_seg_si 0.0 --w_seg "$w_seg" --lr 1.6100e-7 --tags "kitti_baseline,w_seg_sweep"
done