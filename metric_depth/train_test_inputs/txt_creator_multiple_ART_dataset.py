import os
from tqdm import tqdm

# Define dataset path, track, and bag list
dataset_path = "/home/art/Depth-Anything/metric_depth/data/ART/"
track = "IMS"
bags = ["ims_2024_09_04-09_47_56_new_2Hz", "rosbag2_2024_09_04-13_17_48_9"]  # Specify list of bags here
train_eval = "train"  # or "train"

dataset_type = "art"

# Open file for writing combined output
output_path = f"./{dataset_type}_{len(bags)}_bag_{train_eval}_{track}_filenames.txt"
with open(output_path, "w") as f:
    print(f"Writing output to {output_path}...")
    
    # Iterate through each bag in the list
    for bag in bags:
        print(f"Processing bag: {bag}...")
        
        # Construct the dataset path for the current bag
        current_bag_path = os.path.join(dataset_path, track, bag)
        
        # If bag path doesn't exist, raise an error
        if not os.path.exists(current_bag_path):
            print(f"Bag path not found for bag: {bag}. ERROR")
            exit(0)
        
        # Collect image paths
        image_paths = []
        print("Collecting image paths...")
        image_dir = os.path.join(current_bag_path, "image")
        if not os.path.exists(image_dir):
            print(f"Image directory not found for bag: {bag}. Skipping...")
            continue
        for image_path in os.listdir(image_dir):
            image_paths.append(os.path.join(current_bag_path, "image", image_path))
        # Sort image paths to ensure a consistent order
        image_paths.sort()
        print(f"Total images collected for bag {bag}: {len(image_paths)}")

        # Collect calibration paths
        calibration_paths = []
        print("Collecting calibration paths...")
        calibration_dir = os.path.join(current_bag_path, "intrinsics")
        if not os.path.exists(calibration_dir):
            print(f"Calibration directory not found for bag: {bag}. Skipping...")
            continue
        for calibration_path in os.listdir(calibration_dir):
            calibration_paths.append(os.path.join(current_bag_path, "intrinsics", calibration_path))
        print(f"Total calibration files collected for bag {bag}: {len(calibration_paths)}")

        # Ensure there is at least one calibration file
        if not calibration_paths:
            print(f"No calibration files found for bag: {bag}. Skipping...")
            continue
        
        # Iterate through images with tqdm progress bar
        for i, image_path in enumerate(tqdm(image_paths, desc=f"Processing images in {bag}")):
            calibration_path = calibration_paths[0]  # Assuming same calibration for all images
            
            # Read intrinsic matrix from calibration file
            try:
                with open(calibration_path, "r") as f2:
                    line = f2.readlines()[0]
                    focal_data = line.strip().split()  # Split focal length line into components
                    focal = [float(val) for val in focal_data]  # Convert to float values
                    avg_focal = (focal[0] + focal[4]) / 2  # Calculate average focal length
            except (IndexError, ValueError, FileNotFoundError) as e:
                print(f"Error parsing focal length in {calibration_path}: {e}")
                continue  # Skip if focal length parsing fails
            
            # Write formatted data to output file
            f.write(
                f"{bag}/image/{os.path.basename(image_path)} "
                f"{bag}/groundtruth_depth/{os.path.basename(image_path)} "
                f"{avg_focal:.4f}\n"
            )
            print(f"Processed image: {os.path.basename(image_path)} in bag {bag} with focal length {avg_focal:.4f}")

print("Processing complete.")
