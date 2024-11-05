import os
import glob

# KITTI's dataset requires files with lines "<image_path> <depth_path> <focal_length>" indicating datasets.

# Base directories
base_dir = './data/Kitti/depth_selection/val_selection_cropped'
image_dir = os.path.join(base_dir, 'image')
gt_dir = os.path.join(base_dir, 'groundtruth_depth')
intrinsics_dir = os.path.join(base_dir, 'intrinsics')
output_filename = './train_test_inputs/kitti_testing_filenames.txt'

# Focal length (use a default value or adjust as needed)
default_focal_length = 721.5377  # Adjust if you have a specific focal length

# Open the output file
with open(output_filename, 'w') as outfile:
    # List all image files
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    print(f'Found {len(image_files)} images.')

    for img_file in image_files:
        # Extract the filename
        img_filename = os.path.basename(img_file)

        # Relative image path (relative to base directory)
        img_rel_path = os.path.relpath(img_file, start=base_dir)

        # Corresponding depth map filename
        depth_filename = img_filename.replace('image', 'groundtruth_depth', 1)
        depth_file = os.path.join(gt_dir, depth_filename)
        depth_rel_path = os.path.relpath(depth_file, start=base_dir)
        # print(img_file)
        # print(depth_rel_path)
        # break
        # Check if depth file exists
        if not os.path.exists(depth_file):
            print(f'Depth map not found for image {img_filename}, skipping.')
            continue  # Skip if depth map is missing

        # Corresponding intrinsics filename
        intrinsics_filename = img_filename.replace('.png', '.txt')
        intrinsics_file = os.path.join(intrinsics_dir, intrinsics_filename)

        # Get focal length from intrinsics file
        focal_length = default_focal_length
        # if focal_length is None:
        #     focal_length = 707.0912  # Default focal length if intrinsics file is missing

        # Write to output file
        line = f'{img_rel_path} {depth_rel_path} {focal_length}\n'
        outfile.write(line)
        # break
    print("Done Writing File!")
