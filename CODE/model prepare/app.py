# from fastapi import FastAPI, WebSocket
# from fastapi.responses import HTMLResponse
# import numpy as np
# import cv2
# import cv2
# import numpy as np
# import tensorflow as tf
# from collections import deque
# import keras
# from tensorflow.python.client import device_lib
# import cv2
# import numpy as np
# from tensorflow import keras
# from keras.applications.inception_v3 import InceptionV3
# from keras.models import Model
# from collections import deque
# import threading
# import tensorflow as tf
# import os
# import pandas as pd
#
# import cv2
# import numpy as np
# # from skimage.feature import hog
# from tensorflow.keras.models import Model
# from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
# from tensorflow.python.client import device_lib
# from keras.models import load_model
# import keras
# import cv2
# import numpy as np
# # from skimage.feature import hog
# from tensorflow.keras.models import Model
# from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
# from tensorflow.python.client import device_lib
# from keras.models import load_model
# import keras
# from tensorflow import keras
# from keras.applications.inception_v3 import InceptionV3
# from keras.models import Model
# from collections import deque
# import threading
# import asyncio
#
# import base64
# import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# # Create a simple tensor
# a = tf.constant([1.0, 2.0])
#
# # Place the tensor operation on the GPU
# with tf.device('/GPU:0'):
#     b = tf.reduce_sum(a)
#
# print(b.numpy())
#
# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
#
#
# app = FastAPI()
#
# html = """
# <!DOCTYPE html>
# <html>
# <head>
#     <title>Shoplifting Detection</title>
#     <style>
#         canvas {
#             position: absolute;
#             top: 0;
#             left: 0;
#         }
#         video {
#             position: absolute;
#             top: 0;
#             left: 0;
#         }
#     </style>
# </head>
# <body>
#     <h1>Shoplifting Detection Stream</h1>
#     <video id="video" playsInline autoplay></video>
#     <canvas id="canvas"></canvas>
#     <script>
#         var video = document.getElementById('video');
#         var canvas = document.getElementById('canvas');
#         var context = canvas.getContext('2d');
#         var websocket = new WebSocket('ws://' + window.location.host + '/ws');
#
#         websocket.onmessage = function(event) {
#             context.clearRect(0, 0, canvas.width, canvas.height);
#             context.fillText(event.data, 10, 50);
#         };
#
#         if (navigator.mediaDevices.getUserMedia) {
#             navigator.mediaDevices.getUserMedia({ video: true })
#                 .then(function (stream) {
#                     video.srcObject = stream;
#                     video.addEventListener('play', function() {
#                         canvas.width = video.videoWidth;
#                         canvas.height = video.videoHeight;
#                         setInterval(function() {
#                             context.drawImage(video, 0, 0, canvas.width, canvas.height);
#                             if (video.paused || video.ended) return;
#                             var frame = canvas.toDataURL('image/jpeg', 0.5);
#                             websocket.send(frame);
#                         }, 100); // send frame every 100 ms
#                     });
#                 })
#                 .catch(function (error) {
#                     console.log("Something went wrong!");
#                 });
#         }
#     </script>
# </body>
# </html>
# """
#
# @app.get("/")
# async def get():
#     return HTMLResponse(html)
#
# # Load model
# rnn_model = keras.models.load_model('shoplifting_new_method.h5', compile=False)
# base_model = InceptionV3(weights='imagenet', include_top=True)
# feature_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
# frame_buffer = deque(maxlen=145)
#
# @app.websocket("/ws")
# async def webcam_feed_websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#
#     while True:
#         data = await websocket.receive_text()
#         header, encoded = data.split(",", 1)
#         nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
#         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#
#         with tf.device('/GPU:0'):
#             # Handle frame processing
#             if len(frame_buffer) >= 145:
#                 frame_buffer_copy = list(frame_buffer)
#                 batch_frames = [cv2.resize(frame, (299, 299)) for frame in frame_buffer_copy]
#                 batch_frames_np = np.array(batch_frames)
#                 features = feature_model.predict(batch_frames_np)
#                 features = features.reshape(1, 145, -1)
#                 probabilities = rnn_model.predict(features)[0]
#                 print(probabilities)
#                 average_probability = np.mean(probabilities[:, 0])
#
#                 label = "Shoplifting" if average_probability> 0.7 else "Normal"
#                 await websocket.send_text(f"Activity: {label}")
#             else:
#                 await websocket.send_text("Checking")
#
#         frame_buffer.append(frame)
# Import necessary libraries
import numpy as np
import cv2
from collections import deque
import threading
import base64
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model, load_model

# Initialize the FastAPI app
app = FastAPI()

# GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#Define the static HTML content
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Shoplifting Detection</title>
    <style>
        canvas {
            position: absolute;
            top: 0;
            left: 0;
        }
        video {
            position: absolute;
            top: 0;
            left: 0;
        }
    </style>
</head>
<body>
    <h1>Shoplifting Detection Stream</h1>
    <video id="video" playsInline autoplay></video>
    <canvas id="canvas"></canvas>
    <script>
        var video = document.getElementById('video');
        var canvas = document.getElementById('canvas');
        var context = canvas.getContext('2d');
        var websocket = new WebSocket('ws://' + window.location.host + '/ws');

        websocket.onmessage = function(event) {
            context.clearRect(0, 0, canvas.width, canvas.height);
            context.fillText(event.data, 10, 50);
        };

        if (navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(function (stream) {
                    video.srcObject = stream;
                    video.addEventListener('play', function() {
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        setInterval(function() {
                            context.drawImage(video, 0, 0, canvas.width, canvas.height);
                            if (video.paused || video.ended) return;
                            var frame = canvas.toDataURL('image/jpeg', 0.5);
                            websocket.send(frame);
                        }, 100); // send frame every 100 ms
                    });
                })
                .catch(function (error) {
                    console.log("Something went wrong!");
                });
        }
    </script>
</body>
</html>>
"""

# ROOT Endpoint
@app.get("/")
async def get():
    return HTMLResponse(html)

# Load the model
base_model = InceptionV3(weights='imagenet', include_top=True)
feature_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
rnn_model = load_model('shoplifting_new_method.h5', compile=False)

# Frame Buffer
frame_buffer = deque(maxlen=145)

# Prediction Thread
def prediction_thread():
    while True:
        if len(frame_buffer) >= 145:
            with tf.device('/GPU:0'):
                frame_buffer_copy = list(frame_buffer)
                batch_frames = [cv2.resize(frame, (299, 299)) for frame in frame_buffer_copy]
                batch_frames_np = np.array(batch_frames)
                features = feature_model.predict(batch_frames_np)
                features = features.reshape(1, 145, -1)
                probabilities = rnn_model.predict(features)[0]
                print(probabilities)
                average_probability = np.mean(probabilities[:, 0])
                label = "Shoplifting" if average_probability > 0.7 else "Normal"
                print(f"Activity: {label}")
        asyncio.sleep(0.1)

# Start the prediction thread
thread = threading.Thread(target=prediction_thread)
thread.start()

# WebSocket Endpoint for handling real-time video stream
@app.websocket("/ws")
async def webcam_feed_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        header, encoded = data.split(",", 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame_buffer.append(frame)
