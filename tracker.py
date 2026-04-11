"""
tracker.py
My implementation of a sports tracking pipeline using YOLOv8 + ByteTrack.
I tried to keep things simple and modular so it's easy to follow.

What this does:
- Detects players and ball using YOLOv8
- Tracks them with ByteTrack (assigns unique IDs)
- Estimates speed for each player
- Saves screenshots every few seconds
- Generates a heatmap of where players moved
- Exports CSVs for later analysis

Author: Anusha
"""

import cv2
import csv
import time
import argparse
import os
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
from ultralytics import YOLO


# -----------------------------------------------------------
# Class IDs (from COCO dataset)
# 0 = person, 32 = sports ball
# I'm only interested in these two for cricket
# -----------------------------------------------------------
PERSON_CLS = 0
BALL_CLS = 32
TARGET_CLASSES = [PERSON_CLS, BALL_CLS]

# Confidence thresholds — I raised person conf to 0.50 because
# at 0.35 I was getting too many false detections (trees, shadows etc.)
# which caused extra spurious IDs
CONF_PERSON = 0.50
CONF_BALL = 0.30
CONF_GENERAL = 0.40  # this is the floor we pass to YOLO, per-class filter runs after

# IOU threshold — kept low so fast-moving players still match across frames
IOU_THRESHOLD = 0.30

# Drawing settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICKNESS = 2
TEXT_COLOR = (255, 255, 255)
MAX_TRAIL = 60  # how many past positions to draw for the motion trail

# Screenshot every 3 seconds of video time
# I changed this from hardcoded frame numbers to time-based — much cleaner
SCREENSHOT_INTERVAL_SEC = 3.0

# Speed smoothing — average over last 10 frames
# Without this I was getting crazy spikes like 78 km/h for a standing player
# because one noisy detection during a camera pan would mess everything up
SPEED_SMOOTH_FRAMES = 10

# Pixels per metre calibration
# I measured the cricket pitch in the video (pitch = 20.12m = 22 yards)
# It spanned roughly 160 pixels so: 160 / 20.12 ≈ 7.95
# This needs to be updated for different videos or camera angles
PIXELS_PER_METRE = 7.95

DEFAULT_FPS = 25.0

# Scene cut threshold — if consecutive frames differ by more than this on average
# I treat it as a camera cut and reset the tracker
# Tried a few values, 45 worked well without too many false positives
SCENE_CUT_THRESHOLD = 45.0


# -----------------------------------------------------------
# Global stores for trajectories and speed history
# Using defaultdict so I don't have to initialise per track_id
# -----------------------------------------------------------
_trajectories = defaultdict(list)   # track_id -> list of (frame, cx, cy)
_speed_kmh = {}                     # track_id -> latest speed
_speed_history = defaultdict(list)  # track_id -> list of recent speed values


# -----------------------------------------------------------
# Helper: give each track a unique colour based on its ID
# Using HSV so colours are always vivid and spread out
# -----------------------------------------------------------
def get_track_color(track_id):
    hue = (track_id * 37) % 180
    color_hsv = np.uint8([[[hue, 210, 215]]])
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in color_bgr)


# -----------------------------------------------------------
# Speed estimation with rolling average
# The key fix here vs my earlier version:
# Old code used only 5 positions -> one bad frame -> huge spike
# Now I average 10 frames of displacement and also smooth the
# history list, so the value stabilises nicely
# -----------------------------------------------------------
def estimate_speed(track_id, fps):
    trail = _trajectories[track_id]
    if len(trail) < 3:
        return 0.0

    # only use the last SPEED_SMOOTH_FRAMES positions
    recent = trail[-SPEED_SMOOTH_FRAMES:]

    displacements = []
    for i in range(1, len(recent)):
        dx = recent[i][1] - recent[i-1][1]
        dy = recent[i][2] - recent[i-1][2]
        dist = math.hypot(dx, dy)
        displacements.append(dist)

    if not displacements:
        return 0.0

    avg_px_per_frame = sum(displacements) / len(displacements)
    metres_per_frame = avg_px_per_frame / PIXELS_PER_METRE
    raw_kmh = metres_per_frame * fps * 3.6

    # append to history and take rolling average
    hist = _speed_history[track_id]
    hist.append(raw_kmh)
    if len(hist) > SPEED_SMOOTH_FRAMES:
        hist.pop(0)

    smoothed = sum(hist) / len(hist)
    return smoothed


# -----------------------------------------------------------
# Scene cut detection
# Simple but effective — compare mean pixel diff of grayscale frames
# -----------------------------------------------------------
def is_scene_cut(prev_gray, curr_gray):
    if prev_gray is None:
        return False
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(diff.mean()) > SCENE_CUT_THRESHOLD


# -----------------------------------------------------------
# Draw a fading motion trail behind each player
# Older positions are more transparent
# -----------------------------------------------------------
def draw_trail(frame, track_id, frame_idx, cx, cy):
    trail = _trajectories[track_id]
    trail.append((frame_idx, cx, cy))
    if len(trail) > MAX_TRAIL:
        trail.pop(0)

    for i in range(1, len(trail)):
        alpha = int(255 * i / len(trail))
        cv2.line(
            frame,
            (trail[i-1][1], trail[i-1][2]),
            (trail[i][1], trail[i][2]),
            (0, alpha, 255 - alpha),
            1
        )


# -----------------------------------------------------------
# Draw a filled label box above each bounding box
# -----------------------------------------------------------
def draw_label(frame, text, x1, y1, color):
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4), FONT, FONT_SCALE, TEXT_COLOR, 1, cv2.LINE_AA)


# -----------------------------------------------------------
# Overlay the heatmap on a frame (used if --heatmap flag is set)
# -----------------------------------------------------------
def overlay_heatmap_on_frame(frame, heatmap, alpha=0.35):
    if heatmap.max() == 0:
        return frame
    norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)


# -----------------------------------------------------------
# Filter detections by per-class confidence threshold
# YOLO gives all detections >= CONF_GENERAL, we filter stricter here
# -----------------------------------------------------------
def filter_detections(boxes, ids, confs, cls_ids):
    keep = []
    for i, (cls, conf) in enumerate(zip(cls_ids, confs)):
        threshold = CONF_PERSON if cls == PERSON_CLS else CONF_BALL
        if conf >= threshold:
            keep.append(i)

    if not keep:
        return np.array([]), np.array([]), np.array([]), np.array([])

    return boxes[keep], ids[keep], confs[keep], cls_ids[keep]


# -----------------------------------------------------------
# Save a screenshot with a readable filename
# e.g. frame_027_t01m18s.jpg
# -----------------------------------------------------------
def save_screenshot(frame, frame_idx, elapsed_sec, screenshots_dir, video_name):
    minutes = int(elapsed_sec) // 60
    seconds = int(elapsed_sec) % 60
    filename = f"frame_{frame_idx // 30:03d}_t{minutes:02d}m{seconds:02d}s.jpg"
    path = os.path.join(screenshots_dir, filename)
    cv2.imwrite(path, frame)
    print(f"  screenshot saved: {path}")
    return path


# -----------------------------------------------------------
# Build a contact sheet from all screenshots
# Useful for quickly reviewing what the tracker saw at each interval
# -----------------------------------------------------------
def build_contact_sheet(screenshots_dir, output_path, video_name, tracker_name,
                         cols=8, thumb_w=192, thumb_h=108):

    # get all screenshot files sorted by name
    all_files = os.listdir(screenshots_dir)
    shots = sorted([f for f in all_files if f.endswith(".jpg") and f.startswith("frame_")])

    if not shots:
        print("  no screenshots found, skipping contact sheet")
        return

    n = len(shots)
    rows = math.ceil(n / cols)
    PAD = 4
    HEADER = 40

    sheet_w = cols * (thumb_w + PAD) + PAD
    sheet_h = rows * (thumb_h + PAD) + PAD + HEADER
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    title = f"{video_name}  |  {tracker_name}  |  screenshots every {int(SCREENSHOT_INTERVAL_SEC)}s"
    cv2.putText(sheet, title, (PAD, HEADER - 10), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    for idx, fname in enumerate(shots):
        img = cv2.imread(os.path.join(screenshots_dir, fname))
        if img is None:
            continue
        thumb = cv2.resize(img, (thumb_w, thumb_h))

        row = idx // cols
        col = idx % cols
        y0 = HEADER + PAD + row * (thumb_h + PAD)
        x0 = PAD + col * (thumb_w + PAD)
        sheet[y0:y0 + thumb_h, x0:x0 + thumb_w] = thumb

        label = fname.replace("frame_", "").replace(".jpg", "").replace("_", " ")
        cv2.putText(sheet, label, (x0 + 3, y0 + thumb_h - 5),
                    FONT, 0.32, (255, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, sheet)
    print(f"  contact sheet saved: {output_path}  ({n} thumbnails, {cols} cols)")


# -----------------------------------------------------------
# Simple class to track how many unique IDs we've seen
# and how long each one lived (to estimate ID switches)
# -----------------------------------------------------------
class IDMonitor:
    def __init__(self):
        self.all_ids = set()
        self.lifetimes = defaultdict(int)  # track_id -> frame count

    def update(self, active_ids):
        for tid in active_ids:
            self.all_ids.add(tid)
            self.lifetimes[tid] += 1

    def short_lived(self, min_frames=8):
        # tracks seen for fewer than min_frames are probably ID switches
        return sum(1 for v in self.lifetimes.values() if v < min_frames)

    def summary(self):
        total = len(self.all_ids)
        short = self.short_lived()
        return {
            "total_unique_ids": total,
            "short_lived_tracks": short,
            "estimated_true_objects": total - short
        }


# -----------------------------------------------------------
# Main tracking function
# This is where everything comes together
# -----------------------------------------------------------
def run_tracker(source, output_path, model_path="yolov8n.pt",
                tracker_cfg="bytetrack.yaml", frame_skip=1,
                show_preview=False, show_heatmap=False,
                screenshots_dir="screenshots", export_csv=True,
                fps_override=None):

    tracker_name = "BoT-SORT" if "botsort" in tracker_cfg else "ByteTrack"
    video_name = Path(source).stem

    print(f"\n{'='*55}")
    print(f"Tracker    : {tracker_name}")
    print(f"Source     : {source}  ({video_name})")
    print(f"Output     : {output_path}")
    print(f"Screenshots: every {SCREENSHOT_INTERVAL_SEC}s of video time")
    print(f"{'='*55}")

    # clear global state so re-runs don't bleed into each other
    _trajectories.clear()
    _speed_kmh.clear()
    _speed_history.clear()

    model = YOLO(model_path)
    monitor = IDMonitor()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    screenshot_every_n = max(1, int(fps * SCREENSHOT_INTERVAL_SEC))

    print(f"Video      : {width}x{height}  {fps:.1f}fps  {total_frames} frames")
    print(f"Screenshot every {screenshot_every_n} frames (~{SCREENSHOT_INTERVAL_SEC}s)")

    # create output directories
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(screenshots_dir).mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # accumulate heatmap across all frames
    heatmap = np.zeros((height, width), dtype=np.float32)

    count_log = []    # (frame, count) — for graphing later
    speed_log = []    # (frame, track_id, speed_kmh)
    screenshot_paths = []

    frame_idx = 0
    scene_cuts = 0
    start_time = time.time()
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        elapsed_sec = frame_idx / fps

        # --- scene cut check ---
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if is_scene_cut(prev_gray, curr_gray):
            scene_cuts += 1
            print(f"  scene cut detected at frame {frame_idx}  "
                  f"(total cuts: {scene_cuts}) — resetting tracker")
            # calling track with persist=False clears ByteTrack's internal state
            model.track(frame, persist=False, verbose=False)
            _trajectories.clear()
            _speed_history.clear()
        prev_gray = curr_gray

        # skip frames if frame_skip > 1 (just write original frame to output)
        if frame_idx % frame_skip != 0:
            writer.write(frame)
            continue

        # --- run detection + tracking ---
        results = model.track(
            frame,
            persist=True,
            classes=TARGET_CLASSES,
            conf=CONF_GENERAL,
            iou=IOU_THRESHOLD,
            tracker=tracker_cfg,
            verbose=False
        )

        active_count = 0
        active_ids = []

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            # apply per-class confidence filter
            boxes, ids, confs, cls_ids = filter_detections(boxes, ids, confs, cls_ids)
            active_count = len(ids)
            active_ids = ids.tolist()

            for box, track_id, conf, cls_id in zip(boxes, ids, confs, cls_ids):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                color = get_track_color(int(track_id))

                # update heatmap
                cv2.circle(heatmap, (cx, cy), 15, 1.0, -1)

                # draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS)

                # draw trail and estimate speed for players
                draw_trail(frame, int(track_id), frame_idx, cx, cy)

                speed = 0.0
                if cls_id == PERSON_CLS:
                    speed = estimate_speed(int(track_id), fps)
                    _speed_kmh[int(track_id)] = speed
                    speed_log.append((frame_idx, int(track_id), round(speed, 1)))

                # build label text
                if cls_id == BALL_CLS:
                    label = f"ball #{track_id}  {conf:.2f}"
                elif speed > 0:
                    label = f"player #{track_id}  {conf:.2f}  {speed:.1f}km/h"
                else:
                    label = f"player #{track_id}  {conf:.2f}"

                draw_label(frame, label, x1, y1, color)

        monitor.update(active_ids)
        count_log.append((frame_idx, active_count))

        # overlay heatmap on video if flag is set
        if show_heatmap:
            frame = overlay_heatmap_on_frame(frame, heatmap)

        # --- HUD (top-left info overlay) ---
        elapsed_wall = time.time() - start_time
        live_fps = frame_idx / elapsed_wall if elapsed_wall > 0 else 0
        hud = (f"Frame {frame_idx}/{total_frames}  |  {tracker_name}  |  "
               f"FPS {live_fps:.1f}  |  Visible: {active_count}  |  "
               f"IDs: {len(monitor.all_ids)}")
        cv2.putText(frame, hud, (10, 25), FONT, 0.48, (0, 255, 255), 1, cv2.LINE_AA)

        # --- save screenshot every N frames ---
        if frame_idx % screenshot_every_n == 0:
            spath = save_screenshot(frame, frame_idx, elapsed_sec,
                                    screenshots_dir, video_name)
            screenshot_paths.append(spath)

        writer.write(frame)

        if show_preview:
            cv2.imshow("Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # print progress every 30 frames
        if frame_idx % 30 == 0:
            s = monitor.summary()
            print(f"  frame {frame_idx}/{total_frames}  |  "
                  f"visible: {active_count}  |  "
                  f"total IDs: {s['total_unique_ids']}  |  "
                  f"short-lived (switches~): {s['short_lived_tracks']}  |  "
                  f"scene cuts: {scene_cuts}")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # --- save heatmap as image ---
    hm_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
    hm_path = os.path.join(screenshots_dir, f"heatmap_{tracker_name}.jpg")
    cv2.imwrite(hm_path, hm_color)
    print(f"  heatmap: {hm_path}")

    # --- build contact sheet from screenshots ---
    contact_path = os.path.join(str(Path(output_path).parent),
                                f"contact_sheet_{tracker_name}.jpg")
    build_contact_sheet(screenshots_dir, contact_path, video_name, tracker_name)

    # --- export CSVs ---
    out_dir = str(Path(output_path).parent)
    if export_csv:
        count_csv = os.path.join(out_dir, f"count_{tracker_name}.csv")
        with open(count_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "active_count"])
            w.writerows(count_log)
        print(f"  count CSV: {count_csv}")

        speed_csv = os.path.join(out_dir, f"speed_{tracker_name}.csv")
        with open(speed_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "track_id", "speed_kmh"])
            w.writerows(speed_log)
        print(f"  speed CSV: {speed_csv}")

    # --- final summary ---
    total_time = time.time() - start_time
    s = monitor.summary()

    if count_log:
        peak_frame, peak_count = max(count_log, key=lambda x: x[1])
        avg_count = sum(c for _, c in count_log) / len(count_log)
        avg_fps = frame_idx / total_time if total_time > 0 else 0

        print(f"\n=== {tracker_name} Complete ===")
        print(f"Total unique IDs     : {s['total_unique_ids']}")
        print(f"Short-lived tracks   : {s['short_lived_tracks']}  (ID switch proxy)")
        print(f"Est. true objects    : {s['estimated_true_objects']}")
        print(f"Scene cuts detected  : {scene_cuts}")
        print(f"Screenshots taken    : {len(screenshot_paths)}")
        print(f"Peak visible         : {peak_count}  (frame {peak_frame})")
        print(f"Avg visible/frame    : {avg_count:.1f}")
        print(f"Processing speed     : {avg_fps:.1f} fps")
        print(f"Time taken           : {total_time:.1f}s")

        return {
            "tracker": tracker_name,
            "total_unique_ids": s["total_unique_ids"],
            "short_lived_tracks": s["short_lived_tracks"],
            "est_true_objects": s["estimated_true_objects"],
            "scene_cuts": scene_cuts,
            "screenshots": len(screenshot_paths),
            "peak_count": peak_count,
            "avg_count": round(avg_count, 1),
            "processing_fps": round(avg_fps, 1),
            "total_time_s": round(total_time, 1),
        }

    return {}


# -----------------------------------------------------------
# Comparison mode — runs ByteTrack then BoT-SORT on same video
# and prints a side-by-side table of results
# -----------------------------------------------------------
def run_comparison(source, output_dir, model_path="yolov8n.pt"):
    print("\n" + "=" * 60)
    print("TRACKER COMPARISON MODE")
    print("Running ByteTrack first, then BoT-SORT on the same video.")
    print("=" * 60)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    stats_bt = run_tracker(
        source=source,
        output_path=os.path.join(output_dir, "output_bytetrack.mp4"),
        model_path=model_path,
        tracker_cfg="bytetrack.yaml",
        screenshots_dir=os.path.join(output_dir, "shots_bytetrack"),
    )

    stats_bs = run_tracker(
        source=source,
        output_path=os.path.join(output_dir, "output_botsort.mp4"),
        model_path=model_path,
        tracker_cfg="botsort.yaml",
        screenshots_dir=os.path.join(output_dir, "shots_botsort"),
    )

    metrics = [
        ("Total unique IDs",       "total_unique_ids",   "Lower = fewer ID switches"),
        ("Short-lived tracks",     "short_lived_tracks", "Lower = better"),
        ("Est. true objects",      "est_true_objects",   "Should match real player count"),
        ("Scene cuts detected",    "scene_cuts",         "Informational"),
        ("Screenshots taken",      "screenshots",        "Informational"),
        ("Peak detections",        "peak_count",         ""),
        ("Avg detections/frame",   "avg_count",          ""),
        ("Processing speed (fps)", "processing_fps",     "Higher = faster"),
        ("Total time (s)",         "total_time_s",       "Lower = faster"),
    ]

    col_w = 28
    print(f"\n{'Metric':<{col_w}} {'ByteTrack':>12} {'BoT-SORT':>12}  Note")
    print("-" * 80)
    for label, key, note in metrics:
        bt_val = stats_bt.get(key, "N/A")
        bs_val = stats_bs.get(key, "N/A")
        print(f"{label:<{col_w}} {str(bt_val):>12} {str(bs_val):>12}  {note}")
    print("-" * 80)

    # save comparison to CSV
    comp_csv = os.path.join(output_dir, "tracker_comparison.csv")
    with open(comp_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "bytetrack", "botsort", "note"])
        for label, key, note in metrics:
            w.writerow([label, stats_bt.get(key, ""), stats_bs.get(key, ""), note])
    print(f"\nComparison CSV saved: {comp_csv}")

    # simple verdict
    bt_switches = stats_bt.get("short_lived_tracks", 999)
    bs_switches = stats_bs.get("short_lived_tracks", 999)
    winner = "ByteTrack" if bt_switches <= bs_switches else "BoT-SORT"
    print(f"\nVerdict: {winner} produced fewer estimated ID switches on this video.\n")


# -----------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOv8 sports tracker — ByteTrack/BoT-SORT with speed estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source",        default="public_cricket.mp4")
    parser.add_argument("--output",        default="output/tracked_v2.mp4")
    parser.add_argument("--model",         default="yolov8n.pt")
    parser.add_argument("--tracker",       default="bytetrack.yaml",
                        choices=["bytetrack.yaml", "botsort.yaml"])
    parser.add_argument("--skip",          type=int,   default=1,
                        help="Process every Nth frame")
    parser.add_argument("--show",          action="store_true",
                        help="Show live preview window")
    parser.add_argument("--heatmap",       action="store_true",
                        help="Overlay heatmap on output video")
    parser.add_argument("--no-csv",        action="store_true",
                        help="Skip CSV export")
    parser.add_argument("--shots-dir",     default="screenshots")
    parser.add_argument("--fps",           type=float, default=None,
                        help="Override video FPS")
    parser.add_argument("--shot-interval", type=float, default=SCREENSHOT_INTERVAL_SEC,
                        help="Screenshot every N seconds of video time")
    parser.add_argument("--compare",       action="store_true",
                        help="Run both ByteTrack and BoT-SORT and compare")
    parser.add_argument("--compare-out",   default="comparison_output")

    args = parser.parse_args()

    # update the global interval if user passed a different value
    import tracker as _self
    _self.SCREENSHOT_INTERVAL_SEC = args.shot_interval

    if args.compare:
        run_comparison(
            source=args.source,
            output_dir=args.compare_out,
            model_path=args.model,
        )
    else:
        run_tracker(
            source=args.source,
            output_path=args.output,
            model_path=args.model,
            tracker_cfg=args.tracker,
            frame_skip=args.skip,
            show_preview=args.show,
            show_heatmap=args.heatmap,
            screenshots_dir=args.shots_dir,
            export_csv=not args.no_csv,
            fps_override=args.fps,
        )