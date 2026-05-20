import cv2, time, csv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
SOURCE = "output.mp4"   # run on your saved video for reproducible metrics

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise RuntimeError("Cannot open file — run detect.py first")

class_counts = {}
fps_readings = []
total_frames = 0
prev = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t0 = time.time()
    results = model(frame, conf=0.4, verbose=False)[0]
    t1 = time.time()

    fps_readings.append(1 / (t1 - t0))
    total_frames += 1

    for box in results.boxes:
        name = model.names[int(box.cls[0])]
        class_counts[name] = class_counts.get(name, 0) + 1

cap.release()

avg_fps = sum(fps_readings) / len(fps_readings)
unique_classes = len(class_counts)

print(f"\n{'='*40}")
print(f"  Evaluation Results")
print(f"{'='*40}")
print(f"  Total frames processed : {total_frames}")
print(f"  Average FPS            : {avg_fps:.1f}")
print(f"  Min FPS                : {min(fps_readings):.1f}")
print(f"  Max FPS                : {max(fps_readings):.1f}")
print(f"  Unique classes detected: {unique_classes}")
print(f"\n  Class breakdown:")
for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1])[:20]:
    print(f"    {cls:<20} {cnt:>5} detections")

# Save CSV for your report
with open("results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Class", "Detections"])
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        w.writerow([cls, cnt])

print(f"\n  Results saved to results.csv")