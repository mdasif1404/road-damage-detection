from ultralytics import YOLO
import cv2
import os

# Load trained model once
model = YOLO("yolov8_weights/road_damage.pt")

def detect_damage(image_path):
    results = model(image_path, conf=0.10)  # lower confidence

    annotated = results[0].plot()

    output_path = "static/uploads/output.jpg"
    cv2.imwrite(output_path, annotated)

    return "uploads/output.jpg"