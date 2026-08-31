from ultralytics import YOLO

model = YOLO(r"C:\Users\asifm\road_damage_detection\yolov8_weights/road_damage.pt")

results = model(
    "test_images/road2.webp",
    conf=0.10,   # lower confidence threshold
    save=True
)

print("Detection Completed")