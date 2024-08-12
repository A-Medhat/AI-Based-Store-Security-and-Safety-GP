import cv2
from ultralytics import YOLO
import os
from tkinter import messagebox

# detect_dir = r"E:\guipythonproject\runs\detect"
# predictno = len(os.listdir(detect_dir))

# def get_image_path():
#     if predictno == 1:
#         return f"E:\guipythonproject\\runs\detect\predict"
#     else:
#         return f"E:\guipythonproject\\runs\detect\predict{predictno}"

# def detect_live_cam():
#     model_path = "best.pt"
#     model = YOLO(model_path)
#     results =model.predict(source=0, save=False,stream = True,conf=0.4, show=True) 
#     for r in results :
#         predictions = r.tojson(normalize = False)
#         print(predictions)
#         print(type(predictions))

# def detect_image():
#     model_path = "best.pt"
#     model = YOLO(model_path)
#     image_filename = 'zzz.jpg'
#     image_directory = 'templates/pictures'
#     image_path = os.path.join(os.getcwd(), image_directory, image_filename)
#     results = model.predict(source=image_path, save=False, conf=0.4, show=False)
#     for r in results :
#         prediction = r.tojson(normalize = False)
#     print(prediction)
# # detect_live_cam()

# def detect_video(video_path):
#     global predictno  # Access the global predictno variable
#     model_path = "best (1).pt"
#     model = YOLO(model_path)
#     model.predict(source=video_path, save=True, save_dir=get_image_path(), conf=0.4, show=False)  # Save video using dynamic path
#     predictno += 1  # Increase predictno by 1 after each call

#     return get_image_path()  # Return the video path

# Check if any fire detections are present
# fire_detected = any(detection[5] == 0 for detection in detections)  # Assuming fire class index is 0

# if fire_detected:
#     # Show alert message
#     messagebox.showinfo("Fire Detected", "Fire has been detected!")

# cv2.destroyAllWindows()
print("iam"+str(13))