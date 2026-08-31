from ultralytics import YOLO

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")   # nano version (fast & lightweight)

# Train the model
model.train(
    data="data.yaml",
    epochs=20,
    imgsz=416,
    batch=4,
    workers=2,
    device='cpu',
    amp=False
)