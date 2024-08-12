
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, Request, Response, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import tensorflow as tf
from collections import deque
import keras
from tensorflow.python.client import device_lib
from tensorflow import keras
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import load_model
import threading
from fastapi.templating import Jinja2Templates
import base64
import subprocess
from fastapi import HTTPException
import asyncio
import os
import aiofiles
from pydantic import BaseModel
from skimage.feature import hog
import aiofiles
from ultralytics import YOLO

with tf.device('/GPU:0'):
    templates = Jinja2Templates(directory="templates")

    print(device_lib.list_local_devices())
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    async def get(request: Request):
        return templates.TemplateResponse("index.html",  {"request": request})

    app.mount("/templates", StaticFiles(directory="templates"), name="templates")

    @app.get("/about", response_class=HTMLResponse)
    async def about(request: Request):
        return templates.TemplateResponse("about.html", {"request": request})

    ##############################################################################################
    ############################ Live Shoplifting Detection ######################################
    ##############################################################################################

    ################## Model Functions ###################
    # Load the model
    base_model = ""
    feature_model = ""
    rnn_model = ""

    frame_buffer = deque(maxlen=145)

    activity = "Checking..."

    def prediction_thread():
        global base_model
        base_model = InceptionV3(weights='imagenet', include_top=True)
        global feature_model
        feature_model = Model(inputs=base_model.input,
                              outputs=base_model.layers[-2].output)
        global rnn_model
        rnn_model = load_model('shoplifting_new_method0.h5', compile=False)
        while True:
            if len(frame_buffer) >= 145:
                with tf.device('/GPU:0'):
                    frame_buffer_copy = list(frame_buffer)
                    batch_frames = [cv2.resize(frame, (299, 299))
                                    for frame in frame_buffer_copy]
                    batch_frames_np = np.array(batch_frames)
                    features = feature_model.predict(batch_frames_np)
                    features = features.reshape(1, 145, -1)
                    probabilities = rnn_model.predict(features)[0]

                    average_probability = np.mean(probabilities[:, 0])
                    print(average_probability)
                    global activity
                    activity = "Shoplifting" if average_probability > 0.7 else "Normal"
                    print(f"Activity: {activity}")
            else:
                activity = "Checking..."
            asyncio.sleep(0.1)

    # Start the prediction thread
    thread = threading.Thread(target=prediction_thread)
    thread.start()

    ################################################################################################
    ################## Backend Methods ###################

    @app.get("/live-demo", response_class=HTMLResponse)
    async def live_demo(request: Request):
        return templates.TemplateResponse("liveDemo.html", {"request": request})

    @app.websocket("/ws")
    async def webcam_feed_websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        while True:
            data = await websocket.receive_text()
            header, encoded = data.split(",", 1)
            nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame_buffer.append(frame)
            global activity
            if activity != "Checking...":
                await websocket.send_text(f"Activity: {activity}")

    ###################################################################################################
    ########################## The Second Script of the shoplifing detection ##########################
    ###################################################################################################

    ################## Model Functions ###################
    print(device_lib.list_local_devices())

    rnn_model_script = ""
    feature_model_script = ""
    video_path = ''

    def predict_shoplifting(vid_path):
        print("i entered the model")
        global rnn_model
        rnn_model_script = keras.models.load_model(
            'shopliftingmodel.h5', compile=False)
        global feature_model_script
        feature_model_script = keras.models.load_model('mobilnet.h5')

        # Define constants
        FRAMES_PER_VIDEO = 30  # 30 frames per 3-second video
        FRAME_HEIGHT = 224
        FRAME_WIDTH = 224
        global video_path
        video_path = vid_path

        def process_video(video_path, feature_model, rnn_model):
            # Open the input video
            cap = cv2.VideoCapture(video_path)

            # Create the output video
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('output_video.mp4', fourcc,
                                  fps, (frame_width, frame_height))

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
                processed_frame = tf.keras.applications.mobilenet_v3.preprocess_input(
                    resized_frame)
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
                        avg_probability = np.mean(
                            [pred[0] for pred in predictions_window[-frames_per_2_seconds:]])
                        print(avg_probability)
                        # Decide the label based on the average probability threshold
                        label = "Shoplifting" if avg_probability > 0.4 else "Normal"

                        # Draw the decision label on to the frames
                        for i in range(FRAMES_PER_VIDEO):
                            text_label = f"{label}"
                            annotated_frame = batch_frames[i].copy()
                            text_size = cv2.getTextSize(
                                text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                            text_x = frame_width - text_size[0] - 20
                            text_y = 20 + text_size[1]

                            if 'Shoplifting' in text_label:
                                text_color = (0, 0, 255)  # Red color
                            else:
                                text_color = (255, 0, 0)  # Blue color

                            cv2.putText(annotated_frame, text_label, (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                            out_frame = cv2.resize(
                                annotated_frame, (frame_width, frame_height))
                            out.write(out_frame)

                        predictions_window.pop(0)
                    batch_frames.clear()

            cap.release()
            out.release()
            converted_video_path = 'output_video_compatible6.mp4'
            conversion_command = f'ffmpeg -i output_video.mp4 -c:v libx264 -crf 23 -c:a aac -strict -2 {converted_video_path}'

            # Execute the FFmpeg command
            subprocess.run(conversion_command, shell=True)

        process_video(video_path, feature_model_script, rnn_model_script)

    ###############################################################################################################
    ################## Backend Methods ###################

    class VideoUpload(BaseModel):
        video: bytes

    @app.post("/upload-video/")
    async def upload_video_and_run_model(file: UploadFile = File(...)):
        video_storage_path = "templates/videos/detection_output"
        os.makedirs(video_storage_path, exist_ok=True)
        video_path = f"{video_storage_path}/{file.filename}"

        # Asynchronously save the uploaded file
        async with aiofiles.open(video_path, 'wb') as out_file:
            while content := await file.read(1024):
                await out_file.write(content)
        file.file.seek(0)

        # run the model after uploading the video
        await predict_shoplifting(video_path)

        return {"video_path": "/processed-video/"}

    @app.get('processed-video/')
    async def get_processed_video():
        return FileResponse('shoplifting_output.mp4', media_type='video/mp4')
    ###############################################################################################################

    ######################################################################################################
    ######################################### Live Fire Detection ########################################
    ######################################################################################################


    model_path = "best.pt"
    model = YOLO(model_path)


    @app.websocket("/ws/fire_detection")
    async def fire_detection_websocket(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                try:
                    data = await websocket.receive_text()
                    header, encoded = data.split(",", 1)
                    img_arr = np.frombuffer(
                        base64.b64decode(encoded), dtype=np.uint8)
                    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                    results = model.predict(
                        source=img, save=False, conf=0.4, show=False, stream=True)
                    for r in results:
                        prediction = r.tojson(normalize=False)
                        print(prediction)
                    await websocket.send_text("Fire Detected 🔥" if prediction != "[]" else "No Fire Detected")
                except asyncio.CancelledError:
                    break

        except Exception as e:
            print(f"WebSocket error: {e}")
            await websocket.accept()

    if __name__ == "_app_":
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8000)

    ###################################################################################################
    ########################## The Second Script of the fire detection ##########################
    ###################################################################################################

    ################## Model Functions ###################
    #
    detect_dir = r"C:\Users\Dozer\Desktop\GP using FastAPI-last\GP using FastAPI-last\templates\videos\fire_detection"
    predictno = len(os.listdir(detect_dir))

    def get_image_path():
        if predictno == 1:
            return r"C:\Users\Dozer\Desktop\GP using FastAPI-last\GP using FastAPI-last\templates\videos\fire_detection"
        else:
            return r"C:\Users\Dozer\Desktop\GP using FastAPI-last\GP using FastAPI-last\templates\videos\fire_detection\predict{predictno}"

    def detect_video(video_path):
        print("i entered the model")
        model_path = "best.pt"
        model = YOLO(model_path)
        model.predict(source=video_path, save=True,
                      save_dir=get_image_path(), conf=0.4, show=False)
        predictno += 1

    @app.post("/upload-video/fire/")
    async def upload_video_and_run_model(file: UploadFile = File(...)):
        video_storage_path = "templates/videos/fire_detection"
        os.makedirs(video_storage_path, exist_ok=True)
        video_path = f"{video_storage_path}/{file.filename}"

        # Asynchronously save the uploaded file
        async with aiofiles.open(video_path, 'wb') as out_file:
            while content := await file.read(1024):
                await out_file.write(content)
        file.file.seek(0)

        # run the model after uploading the video
        await detect_video(video_path)

        return {"video_path": "/processed-video/fire/"}

    @app.get('/processed-video/fire/')
    async def get_processed_video():
        print('/runs/detect/predict'+str(predictno)+'/0.avi')
        return FileResponse('/runs/detect/predict'+str(predictno)+'/0.avi', media_type='video/mp4')
