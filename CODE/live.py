

from collections import deque
import threading
import cv2
import numpy as np
from skimage.feature import hog
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.python.client import device_lib
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from keras.models import load_model
import keras


rnn_model = keras.models.load_model('shoplifting_new_method0.h5', compile=False)
base_model = InceptionV3(weights='imagenet', include_top=True)
feature_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

rnn_model.run_eagerly = False
feature_model.run_eagerly = False

frame_buffer = deque(maxlen=145)


def capture_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw the current activity label on the frame
        global current_activity_label
        cv2.putText(frame, current_activity_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Real-time Capture', frame)
        frame_buffer.append(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



current_activity_label = "Checking..."

# Function for frame processing
def process_frames():
    global current_activity_label
    while True:
        if len(frame_buffer) >= 145:
            frame_buffer_copy = list(frame_buffer)
            batch_frames = [cv2.resize(frame, (299, 299)) for frame in frame_buffer_copy]
            batch_frames_np = np.array(batch_frames)
            features = feature_model.predict(batch_frames_np)
            features = features.reshape(1, 145, -1)
            probabilities = rnn_model.predict(features)[0]
            average_probability = np.mean(probabilities[:, 0])


            print(average_probability)
            label = "Shoplifting" if average_probability > 0.85 else "Normal"
            current_activity_label = f"Activity: {label}"

capture_thread = threading.Thread(target=capture_frames, args=())
processing_thread = threading.Thread(target=process_frames, args=())

capture_thread.start()
processing_thread.start()

capture_thread.join()
processing_thread.join()


