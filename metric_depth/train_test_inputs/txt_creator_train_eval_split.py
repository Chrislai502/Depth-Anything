import os
import random
from tqdm import tqdm

# Define dataset path and type
dataset_path = "/home/art/Depth-Anything/metric_depth/data/ART/"
track = "IMS"
bag = "rosbag2_2024_09_04-13_17_48_9"
train_eval_ratio = 0.01  # Ratio for eval; train will be the remaining part

dataset_path = os.path.join(dataset_path, track, bag)
dataset_type = "art"

# Set seed for reproducibility
random.seed(42)

# Collect image paths
image_paths = []
print("Collecting image paths...")
for image_path in os.listdir(os.path.join(dataset_path, "image")):
    image_paths.append(os.path.join(dataset_path, "image", image_path))
# Sort image paths to ensure a consistent order
image_paths.sort()
print(f"Total images collected: {len(image_paths)}")

# Collect calibration paths
calibration_paths = []
print("Collecting calibration paths...")
for calibration_path in os.listdir(os.path.join(dataset_path, "intrinsics")):
    calibration_paths.append(os.path.join(dataset_path, "intrinsics", calibration_path))
print(f"Total calibration files collected: {len(calibration_paths)}")

# Split into train and eval sets
num_eval = max(1, int(len(image_paths) * train_eval_ratio))
num_train = len(image_paths) - num_eval

eval_paths = random.sample(image_paths, num_eval)
train_paths = [path for path in image_paths if path not in eval_paths]


def write_output_file(file_paths, dataset_type, train_eval, calibration_path):
    output_path = f"./{dataset_type}_{train_eval}_filenames.txt"
    with open(output_path, "w") as f:
        print(f"Writing output to {output_path}...")
        for image_path in tqdm(file_paths, desc=f"Processing {train_eval} images"):
            # Read intrinsic matrix from calibration file
            with open(calibration_path, "r") as f2:
                line = f2.readlines()[0]
                focal_data = line.strip().split()
                try:
                    focal = [float(val) for val in focal_data]
                    avg_focal = (focal[0] + focal[4]) / 2
                except (IndexError, ValueError) as e:
                    print(f"Error parsing focal length in {calibration_path}: {e}")
                    continue

            f.write(f"image/{os.path.basename(image_path)} groundtruth_depth/{os.path.basename(image_path)} {avg_focal:.4f}\n")
            print(f"Processed image: {os.path.basename(image_path)} with focal length {avg_focal:.4f}")
    print(f"{train_eval.capitalize()} processing complete.")

# Assuming the same calibration file for all images
calibration_path = calibration_paths[1]

# Write train and eval files
write_output_file(train_paths, dataset_type, "train", calibration_path)
write_output_file(eval_paths, dataset_type, "eval", calibration_path)

print("All processing complete.")

# Print out the split counts
print(f"Train samples: {num_train}")
print(f"Eval samples: {num_eval}")