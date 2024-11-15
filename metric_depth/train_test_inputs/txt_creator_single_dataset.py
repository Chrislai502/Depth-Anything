import os
from tqdm import tqdm

# Define dataset path and type
dataset_path = "/home/art/Depth-Anything/metric_depth/data/ART/"
track = "KMS"
bag = "ks_2024_08_24-16_43_10_9"
train_eval = "eval" # or "eval"

dataset_path = os.path.join(dataset_path, track, bag)
dataset_type = "art"

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

# Open file for writing output
output_path = f"./{dataset_type}_{train_eval}_{track}_{bag}_filenames.txt"
with open(output_path, "w") as f:
    print(f"Writing output to {output_path}...")
    
    # Iterate through images with tqdm progress bar
    for i, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
        calibration_path = calibration_paths[1]  # Assuming same calibration for all images
        
        # Read intrinsic matrix from calibration file
        with open(calibration_path, "r") as f2:
            line = f2.readlines()[0]
            focal_data = line.strip().split()  # Split focal length line into components
            try:
                focal = [float(val) for val in focal_data]  # Convert to float values
                avg_focal = (focal[0] + focal[4]) / 2       # Calculate average focal length
            except (IndexError, ValueError) as e:
                print(f"Error parsing focal length in {calibration_path}: {e}")
                # continue  # Skip if focal length parsing fails

        # Write formatted data to output file
        f.write(f"image/{os.path.basename(image_path)} groundtruth_depth/{os.path.basename(image_path)} {avg_focal:.4f}\n")
        print(f"Processed image: {os.path.basename(image_path)} with focal length {avg_focal:.4f}")

print("Processing complete.")
