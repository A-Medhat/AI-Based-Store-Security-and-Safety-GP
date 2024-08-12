import os
import pandas as pd

import cv2
import numpy as np
from skimage.feature import hog
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.python.client import device_lib
from keras.models import load_model
import keras

print(device_lib.list_local_devices())

video_path='Normal_Videos_015_x264.mp4'
model=keras.models.load_model('shoplifting_new_method.h5', compile = False)
base_model = InceptionV3(weights='imagenet', include_top=True)
feature_extraction_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
def optical_flow_planes(prev_frame, next_frame):
    # Convert frames to grayscale for optical flow calculation
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    # Calculate dense optical flow using Farneback's method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Compute the angle and magnitude
    angle, magnitude = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Normalize magnitude to be between 0 and 255
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return angle, magnitude


def hog_plane(frame):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate HOG features
    hog_features, hog_image = hog(gray_image, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    return hog_image


def extract_frames(video_path):
    video = cv2.VideoCapture(video_path)
    frames = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()
    return frames


def fuse_features(rgb_features, composite_features):
    fused_features = (rgb_features + composite_features) / 2
    return fused_features


def extract_features(model, frame):
    frame = np.expand_dims(frame, axis=0)
    features = model.predict(frame)
    return features


def load_and_preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (299, 299))
    frame = preprocess_input(frame)
    return frame


def extract_video_features(video_path, model):
    frames = extract_frames(video_path)
    adjusted_frames = adjust_frames(frames,290)
    features_list = []
    for i in range(1, len(adjusted_frames),2):
        prev_frame = adjusted_frames[i - 1]
        next_frame = adjusted_frames[i]
        rgb_frame = load_and_preprocess_frame(adjusted_frames[i])
        angle, magnitude = optical_flow_planes(prev_frame, next_frame)
        hog_image = hog_plane(next_frame)

        composite_frame = np.dstack((angle, magnitude, hog_image)).astype(np.uint8)

        composite_frame = load_and_preprocess_frame(composite_frame)
        composite_features = extract_features(model, composite_frame)
        rgb_features = extract_features(model, rgb_frame)
        final_features = fuse_features(rgb_features, composite_features)
        final_features = final_features.flatten()

        features_list.append(list(final_features))

    return features_list


def adjust_frames(frames, target_length=290):
    num_frames = len(frames)
    adjusted_frames = frames.copy()

    if num_frames == target_length:
        # If the number of frames already matches the target, return as is
        return adjusted_frames
    elif num_frames < target_length:
        # If fewer frames, repeat the last (target_length - num_frames) frames.
        extra_frames_needed = target_length - num_frames
        repeated_frames = frames[-extra_frames_needed:]
        adjusted_frames.extend(repeated_frames)
    else:  # num_frames > target_length
        # Truncate frames to match the target length
        adjusted_frames = frames[:target_length]

    return adjusted_frames


    return adjusted_frames




#final_features = extract_video_features(video_path, model)
#df = pd.DataFrame(final_features)

# Save DataFrame to CSV
#df.to_csv('video_features.csv', index=False)
def predict_and_visualize(video_path, model, feature_extraction_model):
    # Initialize video capture and writer
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    # Extract and adjust frames
    frames = extract_frames(video_path)  # This should be defined elsewhere
    adjusted_frames = adjust_frames(frames, 290)  # Assuming this adjusts frames to the desired count

    # Feature extraction step
    features_list = extract_video_features(video_path, feature_extraction_model)  # This should be defined elsewhere
    features_list = np.array(features_list)  # Convert to NumPy array and reshape for prediction
    features_list = np.expand_dims(features_list, axis=0)

    # Make prediction
    predictions = model.predict(features_list)

    # Apply labels to each pair of frames based on the prediction
    for i, prediction in enumerate(predictions.flatten()):
        text_label = 'Shoplifting' if prediction > 0.5 else 'Normal Activity'

        # Apply label to two frames since 145 predictions for 290 frames
        for j in range(2):
            frame_index = i * 2 + j
            frame = adjusted_frames[frame_index]
            cv2.putText(frame, text_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)
            out_frame = cv2.resize(frame, (frame_width, frame_height))
            out.write(out_frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


predict_and_visualize(video_path, model, feature_extraction_model)

