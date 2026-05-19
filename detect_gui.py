import cv2
import time
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# ── Config ────────────────────────────────────────────────
SOURCE          = 0        # 0 = webcam  |  "video.mp4" for a file
CONF_THRESHOLD  = 0.40     # starting confidence (slider changes this live)
SAVE_OUTPUT     = True     # saves annotated video to output.mp4
WINDOW_NAME     = "YOLOv8 Object Detection"
PANEL_W         = 320      # width of the right-side stats panel
# ─────────────────────────────────────────────────────────

model      = YOLO("yolov8n.pt")
CLASS_NAMES = model.names  # {0: 'person', 1: 'bicycle', ...}

np.random.seed(42)
COLORS = {i: tuple(int(c) for c in np.random.randint(60, 230, 3))
          for i in range(len(CLASS_NAMES))}

# ── Session state ─────────────────────────────────────────
session_counts  = defaultdict(int)   # lifetime totals per class
session_frames  = 0
fps_history     = []
paused          = False
conf_threshold  = CONF_THRESHOLD

# ── Helpers ───────────────────────────────────────────────
def draw_box(img, x1, y1, x2, y2, label, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    by1, by2 = max(y1 - th - 8, 0), y1
    cv2.rectangle(img, (x1, by1), (x1 + tw + 6, by2), color, -1)
    lum = 0.299*color[2] + 0.587*color[1] + 0.114*color[0]
    txt_color = (0, 0, 0) if lum > 128 else (255, 255, 255)
    cv2.putText(img, label, (x1 + 3, by2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, txt_color, 1, cv2.LINE_AA)


def make_panel(height, top_classes, avg_fps, total_objects,
               unique_classes, total_frames, conf_val):
    """Build the right-side stats panel as a numpy image."""
    panel = np.zeros((height, PANEL_W, 3), dtype=np.uint8)
    panel[:] = (28, 28, 28)

    # thin separator line on left edge
    cv2.line(panel, (0, 0), (0, height), (60, 60, 60), 1)

    y = 20
    def txt(text, x, yy, scale=0.5, color=(200, 200, 200), bold=False):
        t = cv2.FONT_HERSHEY_SIMPLEX
        w = cv2.FONT_HERSHEY_DUPLEX if bold else t
        cv2.putText(panel, text, (x, yy), w, scale, color, 1, cv2.LINE_AA)

    # ── Header ────────────────────────────────────────────
    txt("DETECTION STATS", 12, y, 0.45, (140, 140, 140))
    y += 22

    # ── 4 metric cards in 2x2 grid ───────────────────────
    cards = [
        ("AVG FPS", f"{avg_fps:.1f}",
         (80, 200, 100) if avg_fps >= 15 else (200, 80, 80)),
        ("OBJECTS NOW",  str(total_objects),     (100, 180, 255)),
        ("CLASSES SEEN", f"{unique_classes}/80", (200, 160, 255)),
        ("FRAMES",       str(total_frames),      (180, 180, 180)),
    ]
    cw, ch, pad = (PANEL_W - 36) // 2, 52, 8
    for i, (lbl, val, col) in enumerate(cards):
        cx = pad + i % 2 * (cw + pad)
        cy = y + i // 2 * (ch + pad)
        cv2.rectangle(panel, (cx, cy), (cx + cw, cy + ch), (45, 45, 45), -1)
        cv2.rectangle(panel, (cx, cy), (cx + cw, cy + ch), (60, 60, 60), 1)
        txt(lbl, cx + 6, cy + 14, 0.32, (120, 120, 120))
        txt(val, cx + 6, cy + 38, 0.7, col, bold=True)
    y += 2 * (ch + pad) + 10

    # ── Separator ─────────────────────────────────────────
    cv2.line(panel, (8, y), (PANEL_W - 8, y), (55, 55, 55), 1)
    y += 14

    # ── Top classes bar chart ─────────────────────────────
    txt("TOP CLASSES", 12, y, 0.38, (140, 140, 140))
    y += 16

    if top_classes:
        max_cnt = max(c for _, c in top_classes) or 1
        bar_max = PANEL_W - 110
        for name, count in top_classes[:8]:
            cls_id = next((k for k, v in CLASS_NAMES.items() if v == name), 0)
            col = COLORS[cls_id]
            bar_w = max(4, int(bar_max * count / max_cnt))

            # dot
            cv2.circle(panel, (16, y - 3), 4, col, -1)
            # name
            txt(name[:14], 28, y, 0.42, (210, 210, 210))
            # bar
            bx = PANEL_W - bar_max - 8
            cv2.rectangle(panel, (bx, y - 10), (bx + bar_w, y), col, -1)
            # count
            txt(str(count), bx + bar_w + 4, y, 0.38, (160, 160, 160))
            y += 20

    # ── Separator ─────────────────────────────────────────
    y += 4
    cv2.line(panel, (8, y), (PANEL_W - 8, y), (55, 55, 55), 1)
    y += 14

    # ── Confidence display ────────────────────────────────
    txt("CONFIDENCE THRESHOLD", 12, y, 0.38, (140, 140, 140))
    y += 16
    txt(f"{int(conf_val * 100)}%", 12, y, 0.65, (255, 200, 80), bold=True)
    y += 6

    # confidence bar
    bar_y = y + 6
    cv2.rectangle(panel, (12, bar_y), (PANEL_W - 12, bar_y + 6),
                  (55, 55, 55), -1)
    filled = int((PANEL_W - 24) * conf_val)
    cv2.rectangle(panel, (12, bar_y), (12 + filled, bar_y + 6),
                  (255, 180, 40), -1)
    y += 24

    # ── Controls hint ─────────────────────────────────────
    y = height - 70
    cv2.line(panel, (8, y), (PANEL_W - 8, y), (55, 55, 55), 1)
    y += 14
    hints = [
        "Q — quit",
        "SPACE — pause/resume",
        "+ / - — confidence",
        "S — screenshot",
    ]
    for hint in hints:
        txt(hint, 12, y, 0.37, (100, 100, 100))
        y += 14

    return panel


# ── Open source ───────────────────────────────────────────
cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise RuntimeError("Cannot open video source")

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
src_fps = cap.get(cv2.CAP_PROP_FPS) or 30

writer = None
if SAVE_OUTPUT:
    out_w = frame_w + PANEL_W
    writer = cv2.VideoWriter(
        "output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        src_fps, (out_w, frame_h)
    )

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, frame_w + PANEL_W, frame_h)

prev_time   = time.time()
fps_display = 0.0
frame_count = 0
current_objects = 0
current_classes = {}

print("Running — press Q to quit, SPACE to pause, +/- to adjust confidence")

# ── Main loop ─────────────────────────────────────────────
while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
        print("Paused" if paused else "Resumed")
    elif key == ord('=') or key == ord('+'):
        conf_threshold = min(0.95, conf_threshold + 0.05)
        print(f"Confidence: {conf_threshold:.0%}")
    elif key == ord('-'):
        conf_threshold = max(0.05, conf_threshold - 0.05)
        print(f"Confidence: {conf_threshold:.0%}")
    elif key == ord('s'):
        fname = f"screenshot_{int(time.time())}.jpg"
        cv2.imwrite(fname, frame if 'frame' in dir() else np.zeros((100, 100, 3)))
        print(f"Saved {fname}")

    if paused:
        continue

    ret, frame = cap.read()
    if not ret:
        break

    # ── YOLO inference ────────────────────────────────────
    results = model(frame, conf=conf_threshold, verbose=False)[0]

    frame_detections = {}
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = CLASS_NAMES[cls_id]
        color  = COLORS[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        draw_box(frame, x1, y1, x2, y2, f"{label} {conf:.0%}", color)

        frame_detections[label] = frame_detections.get(label, 0) + 1
        session_counts[label] += 1

    current_objects = sum(frame_detections.values())
    current_classes = frame_detections
    session_frames += 1

    # ── FPS ───────────────────────────────────────────────
    frame_count += 1
    if frame_count % 10 == 0:
        now = time.time()
        fps_display = 10 / max(now - prev_time, 0.001)
        fps_history.append(fps_display)
        prev_time = now

    avg_fps = sum(fps_history[-30:]) / max(len(fps_history[-30:]), 1)

    # ── FPS + object overlay on video ────────────────────
    cv2.putText(frame, f"FPS {fps_display:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (80, 200, 100) if fps_display >= 15 else (80, 80, 200),
                2, cv2.LINE_AA)

    status = "PAUSED" if paused else f"{current_objects} object{'s' if current_objects != 1 else ''}"
    cv2.putText(frame, status, (frame_w - 180, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 180, 255), 2, cv2.LINE_AA)

    # ── Build stats panel ─────────────────────────────────
    top = sorted(session_counts.items(), key=lambda x: -x[1])
    panel = make_panel(
        height        = frame_h,
        top_classes   = top,
        avg_fps       = avg_fps,
        total_objects = current_objects,
        unique_classes= len(session_counts),
        total_frames  = session_frames,
        conf_val      = conf_threshold,
    )

    # ── Stitch video + panel side by side ─────────────────
    combined = np.hstack([frame, panel])

    cv2.imshow(WINDOW_NAME, combined)
    if writer:
        writer.write(combined)

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()

print("\n── Session summary ──────────────────────")
print(f"Frames processed : {session_frames}")
print(f"Avg FPS          : {sum(fps_history)/max(len(fps_history),1):.1f}")
print(f"Unique classes   : {len(session_counts)}")
print("Top 10 classes:")
for name, cnt in sorted(session_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"  {name:<20} {cnt:>5}")
print("Output saved to output.mp4")