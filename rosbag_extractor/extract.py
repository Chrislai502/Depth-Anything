import os
import yaml
import numpy as np
import cv2
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from cv_bridge import CvBridge
from tqdm import tqdm

class ExtractClosestImages:
    def __init__(self, config):
        self.bridge = CvBridge()
        self.config = config

        self.K, self.D = None, None
        self.dataset_path = os.path.join(self.config['out_folder'], self.config['dataset'])
        os.makedirs(self.dataset_path, exist_ok=True)

        # Convert timestamps to integers (since they are stored as nanoseconds)
        self.target_timestamps = {int(ts): dist for ts, dist in self.config['target_timestamps'].items()}
        self.exit_time_tolerance_ns = int(self.config['exit_time_tolerance'] * 1e9)  # Convert 0.5s to nanoseconds

        # Store the closest image for each timestamp, with tracking info
        self.closest_images = {ts: {"image": None, "best_diff": float("inf"), "found": False} for ts in self.target_timestamps}

    def parse_rosbag(self):
        storage_options, converter_options = self.get_rosbag_options_read(self.config['rosbag'])
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}

        metadata = reader.get_metadata()

        with tqdm(total=metadata.message_count, desc="Processing MCAP") as bar:
            while reader.has_next():
                topic, data, t = reader.read_next()
                timestamp_ns = t  # Already in nanoseconds

                # Extract camera intrinsics
                if self.K is None and topic == f'/{self.config["camera_frame"]}/camera_info':
                    msg_type = get_message(type_map[topic])
                    msg = deserialize_message(data, msg_type)
                    self.K = np.array(msg.k).reshape((3, 3))
                    self.D = np.array(msg.d)

                # Extract images and check for closest match
                if topic == f'/{self.config["camera_frame"]}/image':
                    msg_type = get_message(type_map[topic])
                    msg = deserialize_message(data, msg_type)
                    camera_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

                    # Check all target timestamps
                    for target_t, data in self.closest_images.items():
                        if data["found"]:
                            continue  # Skip if already found within 0.5s

                        time_diff = abs(timestamp_ns - target_t)

                        if time_diff < data["best_diff"]:
                            self.closest_images[target_t]["image"] = camera_image
                            self.closest_images[target_t]["best_diff"] = time_diff

                        # If a close enough image (within ±0.5s) is found, save immediately
                        if time_diff <= self.exit_time_tolerance_ns:
                            self.closest_images[target_t]["found"] = True
                            self.save_image(target_t, camera_image)

                bar.update()

                # If all timestamps have been found, exit early
                if all(data["found"] for data in self.closest_images.values()):
                    print("✅ All target timestamps matched within 0.5s. Exiting early.")
                    break

    def save_image(self, target_t, camera_image):
        D = self.target_timestamps[target_t]  # Euclidean distance from config

        # # Compute Z (perpendicular distance)
        # if self.K is not None:
        #     fx = self.K[0, 0]  # Focal length in x
        #     fy = self.K[1, 1]  # Focal length in y
        #     Z = D / np.sqrt(1 + (fx / fy) ** 2)  # Approximate perpendicular Z
        # else:
        #     Z = D  # Fallback if K is not available

        # Save in folder named by Z distance
        save_folder = os.path.join(self.dataset_path, f"{D:.2f}")
        os.makedirs(save_folder, exist_ok=True)

        image_name = f"{target_t}.png"
        cv2.imwrite(os.path.join(save_folder, image_name), camera_image)
        print(f"✅ Saved image for timestamp {target_t} in {save_folder}")


    def get_rosbag_options_read(self, path, serialization_format='cdr'):
        storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='mcap')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format=serialization_format,
            output_serialization_format=serialization_format
        )
        return storage_options, converter_options


if __name__ == "__main__":
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    config['rosbag'] = os.path.normpath(config['rosbag'])
    config['transform']['R'] = np.array(config['transform']['R']).reshape((3, 3))
    config['transform']['T'] = np.array(config['transform']['T'])

    extractor = ExtractClosestImages(config)
    extractor.parse_rosbag()
