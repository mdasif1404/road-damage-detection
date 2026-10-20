from ultralytics import YOLO

model = YOLO("models/road_damage_best.pt")

results = model(
    "val/images/pothole_359.jpg",
    conf=0.25,
    save=True
)

print("Detection Completed")