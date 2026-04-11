<div align="center">

# Multi-Object Detection & Persistent ID Tracking
### in Public Sports / Event Footage

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=flat-square)
![ByteTrack](https://img.shields.io/badge/Tracker-ByteTrack-green?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

**YOLOv8 + ByteTrack / BoT-SORT pipeline for multi-person tracking with motion trails,
movement heatmaps, speed estimation, and frame-level analytics.**

</div>

---

## 📸 Live Demo

<div align="center">

| Close-up Detection | Multi-Player Tracking | Wide Stadium View |
|---|---|---|
| ![close](screenshots/frame_close_batter.jpg) | ![multi](screenshots/frame_multi_player.jpg) | ![wide](screenshots/frame_wide_stadium.jpg) |
| ID 1 — conf 0.90 | Multiple IDs tracked | IDs 121, 124 visible |

</div>

> 📹 Full annotated output video: `output/tracked_v2.mp4` — see `demo/` folder for walkthrough video.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Pipeline Architecture](#pipeline-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Output Examples](#output-examples)
- [Analytics & Graphs](#analytics--graphs)
- [Tracker Comparison](#tracker-comparison)
- [Assumptions & Limitations](#assumptions--limitations)
- [Possible Improvements](#possible-improvements)
- [File Structure](#file-structure)
- [Video Source](#video-source)

---

## 🎯 Overview

This project implements an **end-to-end computer vision pipeline** that:

- Detects **players and sports balls** in every frame using **YOLOv8m**
- Assigns **persistent unique IDs** to each detected subject using **ByteTrack**
- Estimates **player speed in km/h** using pixel displacement across frames
- Visualises **bounding boxes, ID labels, speed, and motion trails** on an annotated output video
- Generates **analytics** — heatmaps, object-count graphs, speed plots, and a summary CSV

The pipeline is designed to remain stable across real-world challenges:
**occlusion**, **rapid movement**, **camera panning**, and **visually similar-looking players**.

---

## ✅ Features

| Feature | Details |
|---|---|
| Object detection | YOLOv8m — COCO pre-trained, class: `person` (conf ≥ 0.35) |
| Multi-object tracking | ByteTrack (default) or BoT-SORT |
| Persistent IDs | Stable across occlusion and re-entry via Kalman filter |
| **Speed estimation** | **Pixel displacement → km/h displayed on each bounding box** |
| Motion trails | Fading colour-matched path per track ID (last 35 frames) |
| Movement heatmap | Cumulative presence density saved as `heatmap.jpg` |
| Object count CSV | Frame-by-frame detection counts |
| Analytics graphs | 4 publication-ready graphs (see below) |
| Auto-screenshots | Every 3 seconds + on every new player ID detected |
| Contact sheet | All screenshots combined in one grid image automatically |
| HUD overlay | Live FPS, frame index, active count, total IDs (2-line, no overflow) |
| Frame skipping | Optional — inference always runs, drawing skipped for speed |

---

## 🏗️ Pipeline Architecture

```
Input video
    │
    ▼
┌─────────────────────────┐
│   YOLOv8m Detector      │  ← persons (conf ≥ 0.35), IOU ≥ 0.45
│   (ultralytics)         │
└────────────┬────────────┘
             │  raw detections (boxes, confidences, class IDs)
             ▼
┌─────────────────────────┐
│  ByteTrack / BoT-SORT   │  ← associates detections → track IDs
│  (persist=True)         │    Kalman filter + Hungarian matching
└────────────┬────────────┘
             │  tracked objects (ID + box + velocity per frame)
             ▼
┌─────────────────────────┐
│   Visualisation         │  ← boxes, labels, speed (km/h), trails, HUD
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Analytics Engine      │  ← heatmap, CSV, graphs, screenshots, contact sheet
└─────────────────────────┘
             │
             ▼
    tracked_output.mp4  +  graphs/  +  screenshots/  +  heatmap.jpg
```

---

## ⚙️ Installation

### Requirements

- Python 3.9+
- CUDA GPU recommended (CPU works — expect ~8–10 FPS)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/sports-tracker.git
cd sports-tracker

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
ultralytics>=8.0.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
pandas>=2.0.0
supervision>=0.18.0
```

> YOLOv8 weights (`yolov8m.pt`) are downloaded automatically on first run.

---

## 🚀 Usage

### Run the tracker

```bash
python tracker.py \
    --source  public_cricket.mp4 \
    --output  output/tracked_output.mp4 \
    --model   yolov8m.pt \
    --tracker bytetrack.yaml
```

### All CLI options

| Argument | Default | Description |
|---|---|---|
| `--source` | `public_cricket.mp4` | Input video path |
| `--output` | `output/tracked_output.mp4` | Annotated output video path |
| `--model` | `yolov8m.pt` | YOLOv8 weights (`n` / `m` / `l` / `x`) |
| `--tracker` | `bytetrack.yaml` | `bytetrack.yaml` or `botsort.yaml` |
| `--skip` | `1` | Draw annotations every N frames (inference always runs) |
| `--show` | off | Show live preview window while processing |
| `--screenshot-dir` | auto | Custom folder for auto-screenshots |

### Generate analytics graphs

```bash
python generate_graphs.py \
    --csv output/count_over_time.csv \
    --out graphs/
```

Produces **4 graphs** in `graphs/`:

| Graph | Description |
|---|---|
| `graph_player_count.png` | Objects detected per frame + smoothed trend line |
| `graph_speed_distribution.png` | Histogram of player speed values (km/h) |
| `graph_speed_timeline.png` | Speed over time per track ID |
| `graph_id_lifetime.png` | How long each track ID was tracked (frames) |

---

## 📊 Output Examples

### Motion-trail annotated frames

<div align="center">

| Trail Tracking | Boundary View | Final Overs |
|---|---|---|
| ![trail](screenshots/frame_trail_motion.jpg) | ![boundary](screenshots/frame_boundary.jpg) | ![final](screenshots/frame_final.jpg) |
| Fading trails per ID | Wide angle detection | Dense tracking |

</div>

### Actual system screenshots (laptop)

<div align="center">

| Batter Close-up | ICC World Cup Wide Shot | Multi-ID Scene |
|---|---|---|
| ![s1](screenshots/demo_laptop_1.jpg) | ![s2](screenshots/demo_laptop_2.jpg) | ![s3](screenshots/demo_laptop_3.jpg) |
| ID 29, conf 0.74 | IDs 121 & 124 tracked | ByteTrack active |

</div>

### Auto-screenshot contact sheet

<div align="center">

![contact](screenshots/contact_sheet.jpg)
*117 auto-screenshots combined — every 3 seconds + on every new player ID*

</div>

---

## 📈 Analytics & Graphs

All graphs are generated automatically from the output CSV using `generate_graphs.py`.

### Graph 1 — Player count per frame
Shows how many subjects were visible frame-by-frame.
The smoothed trend line removes single-frame noise to reveal crowd activity patterns.

### Graph 2 — Speed distribution (km/h)
Histogram of all player speed estimates across the full video.
Calculated using **pixel displacement between consecutive frames × calibration factor**.
Shows the distribution of walking, jogging, and sprinting speeds.

### Graph 3 — Speed timeline
Speed over time per tracked player ID.
Identifies which players were most active and when sprint events occurred.

### Graph 4 — ID lifetime
How many frames each track ID survived before being lost or re-assigned.
Long bars = stable tracking; short bars = brief detection or ID swap.

> All graphs are saved as high-resolution PNG files in `graphs/` and referenced in the technical report.

---

## ⚖️ Tracker Comparison

Both trackers were tested on the same video clip.

| Criterion | ByteTrack | BoT-SORT |
|---|---|---|
| ID stability on occlusion | Good | Very good |
| ID stability on re-entry | Moderate | Better |
| Processing speed | Faster | Slightly slower |
| Memory usage | Lower | Higher |
| Performance on fast motion | Good | Good |
| Implementation complexity | Simple | Moderate |

**Chosen default: ByteTrack**

ByteTrack was selected because it offered the best balance of speed and accuracy for fast-moving cricket footage. Its two-stage matching — associating high-confidence detections first, then recovering low-confidence ones — is particularly effective for motion-blurred frames during batting strokes. BoT-SORT showed slightly better re-identification after full occlusion, which is useful for dense player clusters.

---

## ⚠️ Assumptions & Limitations

### Assumptions

- Input video is standard MP4 at reasonable resolution (480p or above)
- Players wear distinct jerseys — visually identical subjects may cause ID swaps
- Camera is roughly stationary or panning slowly — rapid zoom reduces accuracy
- Speed estimation uses a fixed pixel-per-metre calibration and is approximate

### Limitations

- **High ID count**: 65+ track IDs across a 2-minute video includes re-assignments due to occlusion and re-entry — not all are unique individuals
- **No ReID module**: A player who exits and re-enters the frame after >40 frames will receive a new ID
- **CPU speed**: ~8–10 FPS on CPU; a CUDA GPU provides 40–60 FPS
- **Motion blur**: Confidence drops to ~0.42 during high-speed batting strokes
- **Spectator false positives**: Background crowd near boundaries occasionally triggers detections

---

## 🔧 Possible Improvements

- **ReID embeddings** — integrate appearance features (OSNet) to maintain IDs across full exits and re-entries
- **Top-view projection** — homography transform to bird's-eye court view for tactical analysis
- **Team clustering** — k-means on jersey colour histograms to auto-assign team labels
- **Ball trajectory prediction** — Kalman filter tuned for ballistic motion
- **Evaluation metrics** — HOTA / MOTA / IDF1 against hand-labelled ground truth
- **GPU deployment** — ONNX + TensorRT export for real-time 60 FPS inference

---

## 📁 File Structure

```
sports-tracker/
├── tracker.py                    ← Main tracking pipeline (detection + ByteTrack + analytics)
├── generate_graphs.py            ← Analytics graph generator (4 plots from CSV)
├── requirements.txt              ← Python dependencies
├── README.md                     ← This file
│
├── report/
│   └── Cricket_Tracker_Report.docx   ← Full technical report (10 sections)
│
├── output/
│   ├── tracked_v2.mp4            ← Annotated output video
│   ├── count_ByteTrack.csv       ← Frame-level detection counts
│   ├── speed_ByteTrack.csv       ← Per-frame speed estimates
│   └── summary_report.txt        ← Quick stats summary
│
├── graphs/
│   ├── graph_player_count.png    ← Objects per frame + trend
│   ├── graph_speed_distribution.png  ← Speed histogram (km/h)
│   ├── graph_speed_timeline.png  ← Speed over time per ID
│   └── graph_id_lifetime.png     ← Track ID survival duration
│
└── screenshots/
    ├── frame_close_batter.jpg    ← Close-up detection sample
    ├── frame_multi_player.jpg    ← Multi-player tracking sample
    ├── frame_wide_stadium.jpg    ← Wide-angle stadium sample
    ├── heatmap_ByteTrack.jpg     ← Movement heatmap
    └── contact_sheet.jpg         ← All 117 auto-screenshots in one grid
```

---

## 🎥 Video Source

| Field | Value |
|---|---|
| Platform | YouTube (public) |
| Event | ICC Men's T20 World Cup |
| Video title | *(add your video title here)* |
| URL | *(add your YouTube link here)* |
| License | Public / Creative Commons |
| Duration used | Full video (2 min 17 sec / 3535 frames) |
| Resolution | 640 × 360 @ 30 FPS |

---

## 📌 Assignment Info

| Field | Value |
|---|---|
| Assignment | Multi-Object Detection and Persistent ID Tracking in Public Sports Footage |
| Type | AI / Computer Vision / Data Science |
| Author | Anusha |
| Model | YOLOv8m |
| Tracker | ByteTrack |
| Language | Python 3.10+ |

---

<div align="center">

*Built with YOLOv8 + ByteTrack — ICC T20 Cricket — Computer Vision Assignment*

</div>
