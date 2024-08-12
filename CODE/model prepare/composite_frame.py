import os
import cv2
import numpy as np
from skimage.feature import hog
from tqdm import tqdm


def optical_flow_planes(prev_frame, next_frame):

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    angle, magnitude = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return angle, magnitude


def hog_plane(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hog_features, hog_image = hog(gray_image, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    return hog_image


def process_datasets(dataset_path, output_path, sample_size):
    categories = ["Shoplifting", "Normal"]
    for category in categories:
        category_dir = os.path.join(output_path, category)
        os.makedirs(category_dir, exist_ok=True)

    for category in categories:
        print(f"Processing category: {category}")
        category_path = os.path.join(dataset_path, category)
        video_folders = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]

        video_folders_sample = video_folders[:min(len(video_folders), sample_size)]
        for video_folder in tqdm(video_folders_sample, desc=f"Processing {category} videos"):
            video_path = os.path.join(category_path, video_folder)
            image_files = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
            prev_frame = None
            save_path = os.path.join(output_path, category, video_folder)
            os.makedirs(save_path, exist_ok=True)

            if not image_files:
                print(f"No images found in {video_path}. Skipping...")
                continue

            for i, image_file in enumerate(image_files):
                if i % 2 != 0:
                    continue
                image_path = os.path.join(video_path, image_file)
                next_frame = cv2.imread(image_path)

                if next_frame is None:
                    print(f"Frame {image_file} couldn't be read. Skipping...")
                    continue

                if i == 0:
                    prev_frame = next_frame
                    continue


                angle, magnitude = optical_flow_planes(prev_frame, next_frame)

                hog_img = hog_plane(next_frame)

                composite_frame = np.dstack((angle, magnitude, hog_img)).astype(np.uint8)  # Ensure data type

                if i % 2 == 0:

                    composite_filename = f'composite_{i}.jpg'
                    composite_filepath = os.path.join(save_path, composite_filename)
                    cv2.imwrite(composite_filepath, composite_frame)

                prev_frame = next_frame

                print(f"Processed {composite_filename} in {video_folder}")



dataset_path = 'rgb/Dataset'
output_path = 'NEWcomposite'
sample_size = 30
process_datasets(dataset_path, output_path, sample_size)
