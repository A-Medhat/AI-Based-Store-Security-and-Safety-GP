import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model


def extract_features(model, frame):
    frame = np.expand_dims(frame, axis=0)
    features = model.predict(frame)
    return features


def load_and_preprocess_frame(frame_path):
    frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (299, 299))
    frame = preprocess_input(frame)
    return frame


def fuse_features(rgb_features, composite_features):
    fused_features = (rgb_features + composite_features) / 2
    return fused_features


import os

def process_and_fuse_frames_for_videos(rgb_frames_folder, composite_frames_folder, output_folder, model_weights_path):

    base_model = InceptionV3(weights=model_weights_path, include_top=True)
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    features_list = []

    for category in os.listdir(rgb_frames_folder):
        category_rgb_folder = os.path.join(rgb_frames_folder, category)
        category_composite_folder = os.path.join(composite_frames_folder, category)

        if not (os.path.isdir(category_rgb_folder) and os.path.isdir(category_composite_folder)):
            continue

        label = 1 if 'Shoplifting' in category else 0

        for video_name in os.listdir(category_rgb_folder):
            video_rgb_folder = os.path.join(category_rgb_folder, video_name)
            video_composite_folder = os.path.join(category_composite_folder, video_name)
            print("RGB video", video_rgb_folder, "------Composite video ", video_composite_folder)
            if not (os.path.isdir(video_rgb_folder) and os.path.isdir(video_composite_folder)):
                continue

            rgb_frames = sorted(os.listdir(video_rgb_folder))[:145]
            composite_frames = sorted(os.listdir(video_composite_folder))[:145]

            for rgb_frame_file, composite_frame_file in zip(rgb_frames, composite_frames):
                rgb_frame_path = os.path.join(video_rgb_folder, rgb_frame_file)
                composite_frame_path = os.path.join(video_composite_folder, composite_frame_file)

                # Load and preprocess frames
                rgb_frame = load_and_preprocess_frame(rgb_frame_path)
                composite_frame = load_and_preprocess_frame(composite_frame_path)

                # Extract features
                rgb_features = extract_features(model, rgb_frame)
                composite_features = extract_features(model, composite_frame)

                # Fuse features
                final_features = fuse_features(rgb_features, composite_features)
                final_features = final_features.flatten()

                features_list.append(list(final_features) + [label])

    final_features_df = pd.DataFrame(features_list)

    final_features_df.columns = [f'feature_{i}' for i in range(final_features_df.shape[1] - 1)] + ['label']


    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    csv_output_path = os.path.join(output_folder, 'final_features.csv')
    final_features_df.to_csv(csv_output_path, index=False)



rgb_frames_folder = 'rgb/Dataset'
composite_frames_folder = 'NEWcomposite'
output_folder = 'New_features'
model_weights_path = 'imagenet'
process_and_fuse_frames_for_videos(rgb_frames_folder, composite_frames_folder, output_folder, model_weights_path)