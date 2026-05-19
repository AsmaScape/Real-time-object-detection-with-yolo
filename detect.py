import cv2
import time
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# ── Config ────────────────────────────────────────────────
SOURCE         = 0
CONF_THRESHOLD = 0.4
SAVE_OUTPUT    = True
PANEL_W        = 260
# ─────────────────────────────────────────────────────────

model       = YOLO("yolov8n.pt")
CLASS_NAMES = model.names

np.random.seed(42)
COLORS = {i: tuple(int(c) for c in np.random.randint(50, 230, 3)) for i in range(80)}

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise RuntimeError("Cannot open video source")

w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_src = cap.get(cv2.CAP_PROP_FPS) or 30

writer = None
if SAVE_OUTPUT:
    writer = cv2.VideoWriter(
        "output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_src, (w + PANEL_W, h)
    )

# ── State ─────────────────────────────────────────────────
prev_time      = time.time()
frame_count    = 0
fps_display    = 0.0
conf_threshold = CONF_THRESHOLD
session_counts = defaultdict(int)
paused         = False

print("Q quit | SPACE pause | +/- confidence | S screenshot")

while True:
    key = cv2.waitKey(1) & 0xFF
    if   key == ord('q'):  break
    elif key == ord(' '):  paused = not paused
    elif key in (ord('+'), ord('=')):
        conf_threshold = min(0.95, conf_threshold + 0.05)
        print(f"Conf: {conf_threshold:.0%}")
    elif key == ord('-'):
        conf_threshold = max(0.05, conf_threshold - 0.05)
        print(f"Conf: {conf_threshold:.0%}")
    elif key == ord('s'):
        fname = f"screenshot_{int(time.time())}.jpg"
        cv2.imwrite(fname, combined if 'combined' in dir() else np.zeros((10,10,3)))
        print(f"Saved {fname}")

    if paused:
        continue

    ret, frame = cap.read()
    if not ret:
        break

    # ── YOLO inference (same as your working script) ──────
    results = model(frame, conf=conf_threshold, verbose=False)[0]

    detected_classes = {}

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = CLASS_NAMES[cls_id]
        color  = COLORS[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        lum = 0.299*color[2] + 0.587*color[1] + 0.114*color[0]
        txt_col = (0, 0, 0) if lum > 128 else (255, 255, 255)
        cv2.putText(frame, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_col, 1, cv2.LINE_AA)

        detected_classes[label] = detected_classes.get(label, 0) + 1
        session_counts[label]  += 1

    # ── FPS (unchanged from your script) ──────────────────
    frame_count += 1
    if frame_count % 10 == 0:
        now         = time.time()
        fps_display = 10 / (now - prev_time)
        prev_time   = now

    # ── FPS + object count on video frame ─────────────────
    total = sum(detected_classes.values())
    cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"Objects: {total}", (w - 180, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    # ── Stats panel (plain numpy, no extra calls) ──────────
    panel = np.full((h, PANEL_W, 3), 28, dtype=np.uint8)

    def pt(text, x, y, scale=0.45, color=(200, 200, 200), bold=False):
        font = cv2.FONT_HERSHEY_DUPLEX if bold else cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel, text, (x, y), font, scale, color, 1, cv2.LINE_AA)

    # header
    pt("DETECTION STATS", 10, 22, 0.42, (130, 130, 130))
    cv2.line(panel, (8, 30), (PANEL_W - 8, 30), (55, 55, 55), 1)

    # 4 metric cards
    fps_col = (80, 200, 100) if fps_display >= 15 else (80, 80, 220)
    cards = [
        ("AVG FPS",      f"{fps_display:.1f}", fps_col),
        ("OBJECTS NOW",  str(total),           (100, 180, 255)),
        ("CLASSES SEEN", f"{len(session_counts)}/80", (200, 160, 255)),
        ("FRAMES",       str(frame_count),     (180, 180, 180)),
    ]
    cw, ch, pad = (PANEL_W - 18) // 2, 50, 5
    for i, (lbl, val, col) in enumerate(cards):
        cx = 6  + i % 2 * (cw + pad)
        cy = 40 + i // 2 * (ch + pad)
        cv2.rectangle(panel, (cx, cy), (cx + cw, cy + ch), (45, 45, 45), -1)
        cv2.rectangle(panel, (cx, cy), (cx + cw, cy + ch), (60, 60, 60), 1)
        pt(lbl, cx + 5, cy + 14, 0.30, (110, 110, 110))
        pt(val, cx + 5, cy + 38, 0.65, col, bold=True)

    y = 40 + 2 * (ch + pad) + 10
    cv2.line(panel, (8, y), (PANEL_W - 8, y), (55, 55, 55), 1)
    y += 14

    # top-classes bar chart
    pt("TOP CLASSES", 10, y, 0.37, (130, 130, 130))
    y += 16
    top = sorted(session_counts.items(), key=lambda x: -x[1])[:8]
    max_cnt = max((c for _, c in top), default=1)
    bar_area = PANEL_W - 100
    for name, count in top:
        cls_id = next((k for k, v in CLASS_NAMES.items() if v == name), 0)
        col    = COLORS[cls_id]
        bar_w  = max(3, int(bar_area * count / max_cnt))
        cv2.circle(panel, (14, y - 3), 4, col, -1)
        pt(name[:12], 26, y, 0.38, (210, 210, 210))
        bx = PANEL_W - bar_area - 6
        cv2.rectangle(panel, (bx, y - 9), (bx + bar_w, y + 1), col, -1)
        pt(str(count), bx + bar_w + 4, y, 0.34, (150, 150, 150))
        y += 19

    # confidence bar
    y = max(y + 8, h - 80)
    cv2.line(panel, (8, y), (PANEL_W - 8, y), (55, 55, 55), 1)
    y += 14
    pt(f"CONF: {conf_threshold:.0%}", 10, y, 0.42, (255, 190, 60), bold=True)
    y += 10
    cv2.rectangle(panel, (10, y), (PANEL_W - 10, y + 6), (55, 55, 55), -1)
    filled = int((PANEL_W - 20) * conf_threshold)
    cv2.rectangle(panel, (10, y), (10 + filled, y + 6), (255, 180, 40), -1)

    # key hints
    y = h - 56
    cv2.line(panel, (8, y), (PANEL_W - 8, y), (55, 55, 55), 1)
    for hint in ["Q quit  SPACE pause", "+/- confidence  S save"]:
        y += 14
        pt(hint, 10, y, 0.32, (90, 90, 90))

    # ── Combine & display ──────────────────────────────────
    combined = np.hstack([frame, panel])
    cv2.imshow("YOLOv8 Detection", combined)
    if writer:
        writer.write(combined)

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()

print("\n── Session summary ─────────────────────")
print(f"Frames : {frame_count}")
print(f"FPS    : {fps_display:.1f}")
print(f"Classes: {len(session_counts)}")
for name, cnt in sorted(session_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"  {name:<20} {cnt}")