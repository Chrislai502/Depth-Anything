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

for scale in $(seq 0.45 0.05 0.75); do 
    echo "Running with scale factor: $scale"
    python3 train_mix_infer.py -d kitti --kitti_scale_factor "$scale" --epoch 5
done