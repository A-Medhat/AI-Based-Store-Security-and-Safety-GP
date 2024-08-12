# import cv2
# import numpy as np
# import os
# from skimage.feature import hog
# from keras.models import load_model
# from keras.applications.inception_v3 import preprocess_input
# # Assuming you have the InceptionV3 model and preprocessing given the context of your feature extraction
#
#
# def optical_flow_planes(prev_frame, next_frame):
#     # Convert frames to grayscale for optical flow calculation
#     prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#     next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
#     # Calculate dense optical flow using Farneback's method
#     flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     # Compute the angle and magnitude
#     angle, magnitude = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#     # Normalize magnitude to be between 0 and 255
#     magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
#     return angle, magnitude
#
#
# def hog_plane(frame):
#     # Convert the image to grayscale
#     gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Calculate HOG features
#     hog_features, hog_image = hog(gray_image, pixels_per_cell=(8, 8),
#                                   cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
#     return hog_image
#
# def preprocess_frame(frame):
#
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame = cv2.resize(frame, (299, 299))
#    # frame = cv2.resize(frame, (299, 299))  # Resize the frame to (299, 299) for InceptionV3
#     frame = preprocess_input(frame)
#     return frame
#
# def create_composite_frame(prev_frame, next_frame):
#     # Compute angle and magnitude from optical flow
#     angle, magnitude = optical_flow_planes(prev_frame, next_frame)
#     # Compute HOG plane
#     hog_img = hog_plane(next_frame)
#     # Concatenate angle, magnitude, and HOG plane into a composite frame
#     composite_frame = np.dstack((angle, magnitude, hog_img)).astype(np.uint8)  # Ensure data type
#     return composite_frame  # Placeholder, use actual composite frame generation
#
# def predict_and_annotate(model, video_path, output_video_path):
#     cap = cv2.VideoCapture(video_path)
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'H264'), fps, (width, height))
#
#     success, prev_frame = cap.read()
#     if not success:
#         print("Failed to read video")
#         return
#
#     prev_frame = preprocess_frame(prev_frame)
#
#     while True:
#         success, next_frame = cap.read()
#         if not success:
#             break
#
#         next_frame_pre = preprocess_frame(next_frame)
#         composite_frame = create_composite_frame(prev_frame, next_frame_pre)
#
#         # Expand dimensions to match model input
#         composite_frame = np.expand_dims(composite_frame, axis=0)
#
#         # Predict shoplifting probability
#         probability = model.predict(composite_frame)[0]
#
#         # Annotate and write frame
#         annotated_frame = next_frame.copy()
#         cv2.putText(annotated_frame, f"Shoplifting Probability: {probability:.2f}",
#                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         out.write(annotated_frame)
#
#         prev_frame = next_frame_pre
#
#     cap.release()
#     out.release()
#
# # Load your model
# model = load_model("shoplifting_new_method.h5")
#
# # Example usage
# input_video_path = "video_2024-03-05_22-24-23.mp4"
# output_video_path = "output_video.mp4"
# predict_and_annotate(model, input_video_path, output_video_path)
import cv2
import numpy as np
import tensorflow as tf
import keras
import subprocess
# Define constants
FRAMES_PER_VIDEO = 30  # 30 frames per 3-second video
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
rnn_model = keras.models.load_model('shopliftingmodel (1).h5',compile=False)
feature_model = keras.models.load_model('mobilnet (2).h5')
video_path = 'stock-footage-crime-concept-criminal-thief-shoplifting-from-retail-store-on-security-camera-screen.webm'


def process_video(video_path, feature_model, rnn_model):
    # Open the input video
    cap = cv2.VideoCapture(video_path)

    # Create the output video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))

    # Initialize variables
    frame_count = 0
    batch_frames = []

    # Calculate frame count for 2 seconds.
    frames_per_2_seconds = int(fps * 2)

    # Initialize a list to keep track of the predictions
    predictions_window = []

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:  # When video capture fails or video ends
            break

        if frame is None:  # Add this check to ensure frame is not None
            continue

        # Preprocess the frame
        resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        processed_frame = tf.keras.applications.mobilenet_v3.preprocess_input(resized_frame)
        batch_frames.append(processed_frame)

        # If we have a full batch, predict and reset
        if len(batch_frames) == FRAMES_PER_VIDEO:
            # ... (feature extraction and prediction stay the same)

            # Store predictions in the window
            batch_frames_np = np.array(batch_frames)
            features = feature_model.predict(batch_frames_np)
            features = features.reshape(1, FRAMES_PER_VIDEO, -1)

            # Predict with rnn model
            predictions = rnn_model.predict(features)[0]
            predictions_window.extend(predictions)

            # Now check if we have enough frames to check for a 2-second interval
            if len(predictions_window) >= frames_per_2_seconds:
                # Calculate the average probability over the window
                avg_probability = np.mean([pred[0] for pred in predictions_window[-frames_per_2_seconds:]])
                print(avg_probability)
                # Decide the label based on the average probability threshold
                label = "Shoplifting" if avg_probability > 0.4 else "Normal"

                # Draw the decision label on to the frames
                for i in range(FRAMES_PER_VIDEO):
                    text_label = f"{label}"
                    annotated_frame = batch_frames[i].copy()
                    text_size = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    text_x = frame_width - text_size[0] - 20
                    text_y = 20 + text_size[1]

                    # cv2.putText(annotated_frame, text_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    ##cv2.putText(annotated_frame, text_label,(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0, 0), 2)
                    if 'Shoplifting' in text_label:
                        text_color = (0, 0, 255)  # Red color
                    else:
                        text_color = (255, 0, 0)  # Blue color

                    cv2.putText(annotated_frame, text_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                    # cv2.putText(annotated_frame, text_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 1)
                    out_frame = cv2.resize(annotated_frame, (frame_width, frame_height))
                    out.write(out_frame)

                predictions_window.pop(0)
            batch_frames.clear()

    cap.release()
    out.release()
    converted_video_path = 'output_video_compatible1.mp4'
    conversion_command = f'ffmpeg -i output_video.mp4 -c:v libx264 -crf 23 -c:a aac -strict -2 {converted_video_path}'


    subprocess.run(conversion_command, shell=True)

    return converted_video_path


processed_video_path = process_video(video_path, feature_model, rnn_model)