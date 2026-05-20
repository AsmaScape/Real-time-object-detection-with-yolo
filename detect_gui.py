import cv2
import time
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque

# ── Config ────────────────────────────────────────────────
SOURCE          = 0        # 0 = webcam  |  "video.mp4" for a file
CONF_THRESHOLD  = 0.40
SAVE_OUTPUT     = True
PANEL_W         = 270      # right stats panel width
GRAPH_H         = 120      # height of each graph strip
HISTORY_LEN     = 150      # frames kept in rolling graph buffers
TARGET_FPS      = 15       # reference line drawn on FPS graph
# ─────────────────────────────────────────────────────────

model       = YOLO("yolov8n.pt")
CLASS_NAMES = model.names

np.random.seed(42)
COLORS = {i: tuple(int(c) for c in np.random.randint(60, 230, 3))
          for i in range(len(CLASS_NAMES))}

# ── Rolling history buffers ────────────────────────────────
fps_history     = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
obj_history     = deque([0]   * HISTORY_LEN, maxlen=HISTORY_LEN)

# ── Session state ─────────────────────────────────────────
session_counts  = defaultdict(int)
all_track_ids   = set()
session_frames  = 0
paused          = False
conf_threshold  = CONF_THRESHOLD

prev_time       = time.time()
fps_display     = 0.0
frame_count     = 0

# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def draw_box(img, x1, y1, x2, y2, label, color, track_id=None):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    full_label = f"ID#{track_id} {label}" if track_id is not None else label
    (tw, th), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    by1 = max(y1 - th - 8, 0)
    cv2.rectangle(img, (x1, by1), (x1 + tw + 6, y1), color, -1)
    lum = 0.299*color[2] + 0.587*color[1] + 0.114*color[0]
    txt_color = (0, 0, 0) if lum > 128 else (255, 255, 255)
    cv2.putText(img, full_label, (x1 + 3, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, txt_color, 1, cv2.LINE_AA)


def draw_graph(history, h, w, color, max_val=None, target_val=None, label=""):
    """
    Render a rolling line graph as a numpy image (h × w × 3).
    history   : deque of numeric values
    max_val   : y-axis ceiling (auto if None)
    target_val: optional horizontal dashed reference line
    """
    canvas = np.full((h, w, 3), 28, dtype=np.uint8)   # dark bg

    vals   = list(history)
    hi     = max_val if max_val else (max(vals) if max(vals) > 0 else 1)
    hi     = max(hi, 1)

    # grid lines
    for pct in [0.25, 0.5, 0.75, 1.0]:
        gy = int(h - pct * (h - 20) - 4)
        cv2.line(canvas, (30, gy), (w - 6, gy), (50, 50, 50), 1)
        val_label = f"{int(pct * hi)}"
        cv2.putText(canvas, val_label, (2, gy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

    # target reference line (dashed)
    if target_val is not None:
        ty = int(h - (target_val / hi) * (h - 20) - 4)
        ty = max(4, min(ty, h - 4))
        for x in range(30, w - 6, 10):
            cv2.line(canvas, (x, ty), (min(x + 5, w - 6), ty), (80, 180, 80), 1)

    # filled area under the curve
    n    = len(vals)
    step = (w - 36) / max(n - 1, 1)
    pts  = []
    for i, v in enumerate(vals):
        px = int(30 + i * step)
        py = int(h - (v / hi) * (h - 20) - 4)
        py = max(4, min(py, h - 4))
        pts.append((px, py))

    if len(pts) >= 2:
        # filled polygon
        fill_pts = [pts[0]] + pts + [(pts[-1][0], h - 2), (pts[0][0], h - 2)]
        fill_arr = np.array(fill_pts, dtype=np.int32)
        overlay  = canvas.copy()
        cv2.fillPoly(overlay, [fill_arr], color)
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
        # line
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i + 1], color, 2, cv2.LINE_AA)
        # current value dot
        cv2.circle(canvas, pts[-1], 3, color, -1, cv2.LINE_AA)

    # label bottom-left
    cv2.putText(canvas, label, (32, h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (130, 130, 130), 1)

    return canvas


def make_panel(frame_h, session_counts, fps, total_now,
               unique_ids, total_frames, conf_val):
    panel = np.full((frame_h, PANEL_W, 3), 28, dtype=np.uint8)
    cv2.line(panel, (0, 0), (0, frame_h), (60, 60, 60), 1)

    def pt(text, x, y, scale=0.45, color=(200, 200, 200), bold=False):
        font = cv2.FONT_HERSHEY_DUPLEX if bold else cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel, text, (x, y), font, scale, color, 1, cv2.LINE_AA)

    y = 20
    pt("DETECTION + TRACKING", 10, y, 0.38, (130, 130, 130))
    cv2.line(panel, (8, y + 6), (PANEL_W - 8, y + 6), (55, 55, 55), 1)
    y += 22

    fps_col = (80, 200, 100) if fps >= TARGET_FPS else (80, 80, 220)
    cards = [
        ("AVG FPS",      f"{fps:.1f}",      fps_col),
        ("TRACKED NOW",  str(total_now),    (100, 180, 255)),
        ("UNIQUE IDs",   str(unique_ids),   (200, 160, 255)),
        ("FRAMES",       str(total_frames), (180, 180, 180)),
    ]
    cw, ch, pad = (PANEL_W - 18) // 2, 50, 5
    for i, (lbl, val, col) in enumerate(cards):
        cx = 6  + i % 2 * (cw + pad)
        cy = y  + i // 2 * (ch + pad)
        cv2.rectangle(panel, (cx, cy), (cx + cw, cy + ch), (45, 45, 45), -1)
        cv2.rectangle(panel, (cx, cy), (cx + cw, cy + ch), (60, 60, 60),  1)
        pt(lbl, cx + 5, cy + 14, 0.30, (110, 110, 110))
        pt(val, cx + 5, cy + 38, 0.65, col, bold=True)
    y += 2 * (ch + pad) + 10

    cv2.line(panel, (8, y), (PANEL_W - 8, y), (55, 55, 55), 1)
    y += 14
    pt("TOP CLASSES", 10, y, 0.37, (130, 130, 130))
    y += 16
    top = sorted(session_counts.items(), key=lambda x: -x[1])[:7]
    max_cnt = max((c for _, c in top), default=1)
    bar_area = PANEL_W - 105
    for name, count in top:
        cls_id = next((k for k, v in CLASS_NAMES.items() if v == name), 0)
        col    = COLORS[cls_id]
        bar_w  = max(3, int(bar_area * count / max_cnt))
        cv2.circle(panel, (14, y - 3), 4, col, -1)
        pt(name[:12], 26, y, 0.37, (210, 210, 210))
        bx = PANEL_W - bar_area - 6
        cv2.rectangle(panel, (bx, y - 9), (bx + bar_w, y + 1), col, -1)
        pt(str(count), bx + bar_w + 4, y, 0.33, (150, 150, 150))
        y += 18

    # confidence bar
    y = max(y + 8, frame_h - 55)
    cv2.line(panel, (8, y), (PANEL_W - 8, y), (55, 55, 55), 1)
    y += 14
    pt(f"CONF: {conf_val:.0%}", 10, y, 0.42, (255, 190, 60), bold=True)
    y += 10
    cv2.rectangle(panel, (10, y), (PANEL_W - 10, y + 6), (55, 55, 55), -1)
    filled = int((PANEL_W - 20) * conf_val)
    cv2.rectangle(panel, (10, y), (10 + filled, y + 6), (255, 180, 40), -1)

    # hints
    y = frame_h - 38
    cv2.line(panel, (8, y), (PANEL_W - 8, y), (55, 55, 55), 1)
    y += 12
    for hint in ["Q quit  SPACE pause", "+/- conf  S screenshot"]:
        pt(hint, 10, y, 0.31, (85, 85, 85))
        y += 13

    return panel


def make_graph_strip(frame_w, total_w):
    """
    Build the bottom graph strip: [objects graph | fps graph]
    Width matches the full composite (frame + panel).
    """
    half = total_w // 2

    obj_graph = draw_graph(
        obj_history, GRAPH_H, half,
        color=(100, 160, 255),
        max_val=20,
        label="objects detected / frame"
    )

    fps_graph = draw_graph(
        fps_history, GRAPH_H, total_w - half,
        color=(80, 200, 100),
        max_val=60,
        target_val=TARGET_FPS,
        label=f"FPS  (-- {TARGET_FPS} FPS target)"
    )

    # section headers
    header_h = 18
    for g, lbl in [(obj_graph, "  Objects Over Time"),
                   (fps_graph, "  FPS Over Time")]:
        cv2.rectangle(g, (0, 0), (g.shape[1], header_h), (38, 38, 38), -1)
        cv2.putText(g, lbl, (6, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    strip = np.hstack([obj_graph, fps_graph])

    # divider between the two graphs
    mid = half
    cv2.line(strip, (mid, 0), (mid, GRAPH_H), (60, 60, 60), 1)

    return strip


# ── Open source ───────────────────────────────────────────
cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise RuntimeError("Cannot open video source")

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
src_fps = cap.get(cv2.CAP_PROP_FPS) or 30

total_w = frame_w + PANEL_W          # composite width
total_h = frame_h + GRAPH_H          # composite height (video + graphs)

writer = None
if SAVE_OUTPUT:
    writer = cv2.VideoWriter(
        "output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        src_fps, (total_w, total_h)
    )

cv2.namedWindow("YOLOv8 Detection + Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Detection + Tracking", total_w, total_h)

print("Q quit | SPACE pause | +/- confidence | S screenshot | T toggle tracking")
tracking_on = True

# ═══════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════
while True:
    key = cv2.waitKey(1) & 0xFF

    if   key == ord('q'): break
    elif key == ord(' '): paused = not paused; print("Paused" if paused else "Resumed")
    elif key in (ord('+'), ord('=')):
        conf_threshold = min(0.95, conf_threshold + 0.05)
        print(f"Conf: {conf_threshold:.0%}")
    elif key == ord('-'):
        conf_threshold = max(0.05, conf_threshold - 0.05)
        print(f"Conf: {conf_threshold:.0%}")
    elif key == ord('s'):
        fname = f"screenshot_{int(time.time())}.jpg"
        if 'composite' in dir(): cv2.imwrite(fname, composite)
        print(f"Saved {fname}")
    elif key == ord('t'):
        tracking_on = not tracking_on
        print(f"Tracking {'ON' if tracking_on else 'OFF'}")

    if paused:
        continue

    ret, frame = cap.read()
    if not ret:
        break

    # ── Inference (tracking or plain detection) ────────────
    if tracking_on:
        results = model.track(
            frame,
            conf=conf_threshold,
            persist=True,        # keeps track IDs stable across frames
            tracker="bytetrack.yaml",
            verbose=False
        )[0]
    else:
        results = model(frame, conf=conf_threshold, verbose=False)[0]

    detected_classes = {}
    current_tracked  = 0

    for box in results.boxes:
        cls_id   = int(box.cls[0])
        conf_val = float(box.conf[0])
        label    = CLASS_NAMES[cls_id]
        color    = COLORS[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # track ID (None if tracking disabled or not yet assigned)
        track_id = None
        if tracking_on and box.id is not None:
            track_id = int(box.id[0])
            all_track_ids.add(track_id)
            current_tracked += 1

        draw_box(frame, x1, y1, x2, y2,
                 f"{label} {conf_val:.0%}", color, track_id)

        detected_classes[label] = detected_classes.get(label, 0) + 1
        session_counts[label]  += 1

    total_now = sum(detected_classes.values())

    # ── FPS ───────────────────────────────────────────────
    frame_count   += 1
    session_frames += 1
    if frame_count % 10 == 0:
        now         = time.time()
        fps_display = 10 / max(now - prev_time, 0.001)
        prev_time   = now

    # update rolling buffers
    fps_history.append(fps_display)
    obj_history.append(total_now)

    avg_fps = sum(list(fps_history)[-30:]) / 30

    # ── Video-frame overlays ──────────────────────────────
    fps_col = (80, 200, 100) if fps_display >= TARGET_FPS else (80, 80, 220)
    cv2.putText(frame, f"FPS {fps_display:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_col, 2, cv2.LINE_AA)

    track_label = (f"{current_tracked} tracked"
                   if tracking_on else f"{total_now} objects")
    cv2.putText(frame, track_label, (frame_w - 190, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 180, 255), 2, cv2.LINE_AA)

    if tracking_on:
        cv2.putText(frame, f"ByteTrack | {len(all_track_ids)} unique IDs",
                    (10, frame_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 140, 220), 1, cv2.LINE_AA)

    # ── Build panel ───────────────────────────────────────
    panel = make_panel(
        frame_h, session_counts, avg_fps,
        current_tracked if tracking_on else total_now,
        len(all_track_ids), session_frames, conf_threshold
    )

    # ── Build graph strip ─────────────────────────────────
    graph_strip = make_graph_strip(frame_w, total_w)

    # ── Composite: [video | panel] on top, [graphs] below ─
    top_row  = np.hstack([frame, panel])
    composite = np.vstack([top_row, graph_strip])

    cv2.imshow("YOLOv8 Detection + Tracking", composite)
    if writer:
        writer.write(composite)

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()

# ── Session summary ───────────────────────────────────────
print("\n── Session summary ──────────────────────────────")
print(f"Frames processed : {session_frames}")
print(f"Avg FPS          : {sum(fps_history)/len(fps_history):.1f}")
print(f"Unique track IDs : {len(all_track_ids)}")
print(f"Unique classes   : {len(session_counts)}")
print("Top 10 classes:")
for name, cnt in sorted(session_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"  {name:<20} {cnt:>5}")
print("Output saved to output.mp4")