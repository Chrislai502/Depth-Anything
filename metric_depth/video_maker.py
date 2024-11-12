import cv2
import os
import glob
import re

def extract_number(filename):
    """Extract number from filename like '1_delta_img.png' -> 1."""
    match = re.search(r'(\d+)_delta_img\.png', filename)
    return int(match.group(1)) if match else float('inf')

def images_to_video(input_folder, output_video_path, fps=30):
    # Get list of image files in the input folder
    images = glob.glob(os.path.join(input_folder, '*_delta_img.png'))
    
    # Sort images by the numeric part of the file name
    images.sort(key=extract_number)

    if not images:
        print("No images found in the folder.")
        return

    # Read the first image to get the dimensions
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    size = (width, height)

    # Define the video codec and create VideoWriter object
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for image_file in images:
        img = cv2.imread(image_file)
        if img is None:
            print(f"Error reading image {image_file}")
            continue
        out.write(img)  # Write the image as a frame to the video

    # Release everything if job is finished
    out.release()
    print(f"Video saved as {output_video_path}")

# Usage example:
input_folder = './output/art'
output_video_path = 'output_video.mp4'
fps = 10  # Frames per second

images_to_video(input_folder, output_video_path, fps)
