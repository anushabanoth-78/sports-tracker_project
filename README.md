<div align="center">

# 🏏 Multi-Object Detection & Persistent ID Tracking
### in Public Sports / Event Footage

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=flat-square)
![ByteTrack](https://img.shields.io/badge/Tracker-ByteTrack-green?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

**YOLOv8 + ByteTrack pipeline for multi-person tracking with motion trails, movement heatmaps, speed estimation, and frame-level analytics.**

</div>

---
## 🎬 Live Demo

> 📹 **[Click here to watch the full demo video](https://drive.google.com/file/d/1Gs_G_2aswDxPRhc6VqKNHUhso3MaJcHE/view?usp=drive_link)**

The demo video shows the complete pipeline running on a **2 min 17 sec ICC Men's T20 World Cup** clip.

### What you'll see in the demo:

| Timestamp | What's Happening |
|---|---|
| 0:00 – 0:20 | YOLOv8 detecting players frame-by-frame with confidence scores |
| 0:20 – 0:50 | ByteTrack assigning persistent IDs with colour-coded bounding boxes |
| 0:50 – 1:20 | Motion trails showing player movement paths (last 35 frames) |
| 1:20 – 1:50 | Speed estimation (km/h) displayed live on each player |
| 1:50 – 2:17 | HUD overlay — live FPS, frame index, active count, total IDs |

### Key highlights:
- ✅ **65+ unique track IDs** assigned across the full video
- ✅ **Stable tracking** through occlusion and camera panning
- ✅ **Real-time speed** displayed per player in km/h
- ✅ **Fading motion trails** per track ID (colour-matched)
- ✅ **Auto-screenshots** captured every 3 seconds + on new ID events

> ⚠️ GitHub does not support video preview for large files.
> Full annotated video also available at: `output/tracked_v2.mp4`

## 📸 Sample Frames

<div align="center">

| Close-up Detection | Multi-Player Tracking | Wide Stadium View |
|---|---|---|
| ![close](https://github.com/anushabanoth-78/sports-tracker_project/blob/main/screenshots/frame_021_t00m21s.jpg) | ![multi](https://github.com/anushabanoth-78/sports-tracker_project/blob/main/screenshots/frame_084_t01m24s.jpg) | ![wide](https://github.com/anushabanoth-78/sports-tracker_project/blob/main/screenshots/frame_108_t01m48s.jpg) |
| ID 1 — conf 0.90 | Multiple IDs tracked | IDs 121, 124 visible |

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Pipeline Architecture](#-pipeline-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Output Files](#-output-files)
- [Analytics & Graphs](#-analytics--graphs)
- [Tracker Comparison](#-tracker-comparison)
- [Assumptions & Limitations](#-assumptions--limitations)
- [Possible Improvements](#-possible-improvements)
- [File Structure](#-file-structure)
- [Video Source](#-video-source)

---

## 🎯 Overview

This project implements an **end-to-end computer vision pipeline** that:

- Detects **players and sports balls** in every frame using **YOLOv8**
- Assigns **persistent unique IDs** to each detected subject using **ByteTrack**
- Estimates **player speed in km/h** using pixel displacement across frames
- Visualises **bounding boxes, ID labels, speed, and motion trails** on an annotated output video
- Generates **analytics** — heatmaps, object-count graphs, speed plots, and summary CSVs

The pipeline is designed to remain stable across real-world challenges:
**occlusion**, **rapid movement**, **camera panning**, and **visually similar-looking players**.

---

## ✅ Features

| Feature | Details |
|---|---|
| Object detection | YOLOv8 — COCO pre-trained, class: `person` (conf ≥ 0.35) |
| Multi-object tracking | ByteTrack (default) or BoT-SORT |
| Persistent IDs | Stable across occlusion and re-entry via Kalman filter |
| Speed estimation | Pixel displacement → km/h displayed on each bounding box |
| Motion trails | Fading colour-matched path per track ID (last 35 frames) |
| Movement heatmap | Cumulative presence density saved as `heatmap.jpg` |
| Object count CSV | Frame-by-frame detection counts |
| Analytics graphs | 4 publication-ready graphs |
| Auto-screenshots | Every 3 seconds + on every new player ID detected |
| Contact sheet | All screenshots combined in one grid image automatically |
| HUD overlay | Live FPS, frame index, active count, total IDs |

---

## 🏗️ Pipeline Architecture

```
Input Video
    │
    ▼
YOLOv8 Detection (per frame)
    │
    ▼
ByteTrack / BoT-SORT (ID assignment)
    │
    ├──► Annotated Output Video (bounding boxes, trails, speed)
    ├──► Screenshots (every 3s + new ID events)
    ├──► Heatmap (cumulative player positions)
    ├──► CSVs (player count + speed per frame)
    └──► graphs.py ──► 4 Analytics Graphs + Summary Report
```

---

## ⚙️ Installation

### Requirements

- Python 3.9+
- CUDA GPU recommended (CPU works — expect ~8–10 FPS)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/anushabanoth-78/sports-tracker_project.git
cd sports-tracker_project

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

> YOLOv8 weights (`yolov8n.pt`) are downloaded automatically on first run.

---

## 🚀 Usage

### Run the tracker

```bash
python tracker.py \
    --source  public_cricket.mp4 \
    --output  output/tracked_output.mp4 \
    --model   yolov8n.pt \
    --tracker bytetrack.yaml
```

### CLI Options

| Argument | Default | Description |
|---|---|---|
| `--source` | `public_cricket.mp4` | Input video path |
| `--output` | `output/tracked_output.mp4` | Annotated output video path |
| `--model` | `yolov8n.pt` | YOLOv8 weights (`n` / `m` / `l` / `x`) |
| `--tracker` | `bytetrack.yaml` | `bytetrack.yaml` or `botsort.yaml` |
| `--skip` | `1` | Draw annotations every N frames (inference always runs) |
| `--show` | off | Show live preview window while processing |

### Generate Analysis Graphs

```bash
python graphs.py --out-dir output/ --fps 30.0
```

---

## 📂 Output Files

| File | Description |
|---|---|
| `output/tracked_v2.mp4` | Annotated output video |
| `output/count_ByteTrack.csv` | Player count per frame |
| `output/speed_ByteTrack.csv` | Per-player speed per frame |
| `output/contact_sheet_ByteTrack.jpg` | All screenshots in one grid |
| `output/summary_report.txt` | Full analysis report |
| `output/graph_player_count.png` | Objects detected per frame |
| `output/graph_speed_distribution.png` | Speed histogram (km/h) |
| `output/graph_speed_timeline.png` | Speed over time per player |
| `output/graph_id_lifetime.png` | Track ID stability chart |
| `screenshots/heatmap_ByteTrack.jpg` | Player position heatmap |

> ⚠️ `tracked_v2.mp4` is 45 MB — GitHub cannot preview it. Use the download button to view locally.

---

## 📊 Output Examples

### Contact Sheet
<div align="center">

![contact](https://github.com/anushabanoth-78/sports-tracker_project/blob/main/output/contact_sheet_ByteTrack.jpg)

*117 auto-screenshots combined — every 3 seconds + on every new player ID*

</div>

### Heatmap
<div align="center">

![heatmap](https://github.com/anushabanoth-78/sports-tracker_project/blob/main/screenshots/heatmap_ByteTrack.jpg)

</div>
## 🎥 Full Output Video

[Download Full Video](https://drive.google.com/uc?export=download&id=1Gs_G_2aswDxPRhc6VqKNHUhso3MaJcHE)

> ⚠️ Large file (~45MB). Click to download directly.

---

## 📈 Analytics & Graphs

### Player Count per Frame
![Player Count](https://github.com/anushabanoth-78/sports-tracker_project/blob/main/output/graph_player_count.png)

### Speed Distribution (km/h)
![Speed Distribution](https://github.com/anushabanoth-78/sports-tracker_project/blob/main/output/graph_speed_distribution.png)

### Speed Timeline
![Speed Timeline](https://github.com/anushabanoth-78/sports-tracker_project/blob/main/output/graph_speed_timeline.png)

### ID Lifetime
![ID Lifetime](https://github.com/anushabanoth-78/sports-tracker_project/blob/main/output/graph_id_lifetime.png)

> 📄 [View Full Summary Report](https://github.com/anushabanoth-78/sports-tracker_project/blob/main/output/summary_report.txt)

---

## ⚖️ Tracker Comparison

| Criterion | ByteTrack | BoT-SORT |
|---|---|---|
| ID stability on occlusion | Good | Very good |
| ID stability on re-entry | Moderate | Better |
| Processing speed | Faster | Slightly slower |
| Memory usage | Lower | Higher |
| Performance on fast motion | Good | Good |

**Chosen default: ByteTrack**

ByteTrack was selected for its best balance of speed and accuracy on fast-moving cricket footage. Its two-stage matching — high-confidence detections first, then recovering low-confidence ones — handles motion-blurred frames effectively.

---

## ⚠️ Assumptions & Limitations

### Assumptions
- Input video is standard MP4 at 480p or above
- Camera is roughly stationary or panning slowly
- Speed estimation uses a fixed pixel-per-metre calibration (approximate)

### Limitations
- **High ID count**: 65+ track IDs across a 2-minute video includes re-assignments due to occlusion — not all are unique individuals
- **No ReID module**: A player who exits and re-enters after >40 frames receives a new ID
- **CPU speed**: ~8–10 FPS on CPU; CUDA GPU provides 40–60 FPS
- **Motion blur**: Confidence drops during high-speed batting strokes
- **Spectator false positives**: Background crowd near boundaries may trigger detections

---

## 🔧 Possible Improvements

- **ReID embeddings** — integrate OSNet appearance features to maintain IDs across re-entries
- **Top-view projection** — homography transform to bird's-eye view for tactical analysis
- **Team clustering** — k-means on jersey colour to auto-assign team labels
- **Ball trajectory prediction** — Kalman filter tuned for ballistic motion
- **Evaluation metrics** — HOTA / MOTA / IDF1 against labelled ground truth
- **GPU deployment** — ONNX + TensorRT export for real-time 60 FPS inference

---

## 📁 File Structure

```
sports-tracker/
│
├── tracker.py                  # Main tracking pipeline (YOLOv8 + ByteTrack)
├── graphs.py                   # Analytics graphs + summary report generator
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── Cricket_Tracker_Report.docx # Technical report
├── yolov8n.pt                  # YOLOv8 model weights
│
├── output/
│   ├── tracked_v2.mp4                  # Annotated output video
│   ├── count_ByteTrack.csv             # Player count per frame
│   ├── speed_ByteTrack.csv             # Per-player speed per frame
│   ├── contact_sheet_ByteTrack.jpg     # All screenshots in one grid
│   ├── graph_player_count.png          # Player count over time
│   ├── graph_speed_distribution.png    # Speed histogram
│   ├── graph_speed_timeline.png        # Speed timeline per player
│   ├── graph_id_lifetime.png           # Track ID lifetime chart
│   └── summary_report.txt             # Full analysis report
│
└── screenshots/
    ├── frame_003_t00m03s.jpg   # Screenshot at 3s
    ├── frame_006_t00m06s.jpg   # Screenshot at 6s
    ├── ...                     # Every 3 seconds of video
    └── heatmap_ByteTrack.jpg   # Player position heatmap
```

---

## 🎥 Video Source

| Field | Value |
|---|---|
| Platform | YouTube (public) |
| Event | ICC Men's T20 World Cup |
| URL | https://www.youtube.com/watch?v=KvSriYbrGD8 |
| License | Public / Creative Commons |
| Duration used | 2 min 17 sec (3535 frames) |
| Resolution | 640 × 360 @ 30 FPS |

---

## 📌 Assignment Info

| Field | Value |
|---|---|
| Assignment | Multi-Object Detection and Persistent ID Tracking in Public Sports Footage |
| Author | Banoth Anusha |
| Institute | IIT Goa |
| Model | YOLOv8n |
| Tracker | ByteTrack |
| Language | Python 3.10+ |

---

<div align="center">

*Built with YOLOv8 + ByteTrack — ICC T20 Cricket — Computer Vision Assignment*

🔗 [GitHub](https://github.com/anushabanoth-78) &nbsp;|&nbsp; 📧 banoth.anusha.22031@iitgoa.ac.in

</div>


