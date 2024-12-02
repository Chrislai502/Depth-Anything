#!/bin/bash

for scale in $(seq 0.51 0.01 0.69); do 
    echo "Running with scale factor: $scale"
    python3 train_mix_infer.py -d mix --kitti_scale_factor "$scale"
done