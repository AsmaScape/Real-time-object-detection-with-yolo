from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # downloads automatically ~6MB
print("Classes available:", len(model.names))
print("Sample classes:", list(model.names.values())[:10])