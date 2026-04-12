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

## 📋 Table of Contents

- [Overview](#-overview)
- [Demo Video](#-demo-video)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Training the Model](#training-the-model)
- [Evaluation & Charts](#-evaluation--charts)
- [How It Works](#-how-it-works)
- [Running the Tracker](#-running-the-tracker)
- [Code Modules](#-code-modules)
- [Technologies Used](#technologies-used)
- [Future Improvements](#-future-improvements)
- [Project Author](#-project-author)

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
⬆️ [Back to Top](#table-of-contents)

## 🎬 Demo Video

> 📹 **[▶ Click here to watch the full demo video](https://drive.google.com/file/d/1Gs_G_2aswDxPRhc6VqKNHUhso3MaJcHE/view?usp=drive_link)**

The demo shows the complete pipeline running on a **2 min 17 sec ICC Men's T20 World Cup** clip.

| Timestamp | What's Happening |
|---|---|
| 0:00 – 0:20 | YOLOv8 detecting players frame-by-frame with confidence scores |
| 0:20 – 0:50 | ByteTrack assigning persistent IDs with colour-coded bounding boxes |
| 0:50 – 1:20 | Motion trails showing player movement paths (last 35 frames) |
| 1:20 – 1:50 | Speed estimation (km/h) displayed live on each player |
| 1:50 – 2:17 | HUD overlay — live FPS, frame index, active count, total IDs |

**Key highlights:**
- ✅ **65+ unique track IDs** assigned across the full video
- ✅ **Stable tracking** through occlusion and camera panning
- ✅ **Real-time speed** displayed per player in km/h
- ✅ **Fading motion trails** per track ID (colour-matched)
- ✅ **Auto-screenshots** captured every 3 seconds + on new ID events

### 📸 Sample Frames

<div align="center">

| Close-up Detection | Multi-Player Tracking | Wide Stadium View |
|---|---|---|
| ![close](https://raw.githubusercontent.com/anushabanoth-78/sports-tracker_project/main/screenshots/frame_021_t00m21s.jpg) | ![multi](https://raw.githubusercontent.com/anushabanoth-78/sports-tracker_project/main/screenshots/frame_084_t01m24s.jpg) | ![wide](https://raw.githubusercontent.com/anushabanoth-78/sports-tracker_project/main/screenshots/frame_108_t01m48s.jpg) |
| ID 1 — conf 0.90 | Multiple IDs tracked | IDs 121, 124 visible |

</div>

### 🎥 More Sample Frames

<div align="center">

| Trail Tracking | Boundary View | Heatmap |
|---|---|---|
| ![trail](https://raw.githubusercontent.com/anushabanoth-78/sports-tracker_project/main/screenshots/frame_039_t00m39s.jpg) | ![boundary](https://raw.githubusercontent.com/anushabanoth-78/sports-tracker_project/main/screenshots/frame_060_t01m00s.jpg) | ![heatmap](https://raw.githubusercontent.com/anushabanoth-78/sports-tracker_project/main/screenshots/heatmap_ByteTrack.jpg) |
| Fading trails per ID | Wide angle detection | Dense tracking heatmap |

</div>

<div align="center">

| Batter Close-up | ICC World Cup Wide Shot | Multi-ID Scene |
|---|---|---|
| ![s1](https://raw.githubusercontent.com/anushabanoth-78/sports-tracker_project/main/screenshots/frame_045_t00m45s.jpg) | ![s2](https://raw.githubusercontent.com/anushabanoth-78/sports-tracker_project/main/screenshots/frame_060_t01m00s.jpg) | ![s3](https://raw.githubusercontent.com/anushabanoth-78/sports-tracker_project/main/screenshots/frame_093_t01m33s.jpg) |
| ID 29, conf 0.74 | IDs 121 & 124 tracked | ByteTrack active |

</div>

> ⚠️ Full annotated video also available at: `output/tracked_v2.mp4` (45 MB — download to view locally)

---
⬆️ [Back to Top](#table-of-contents)

## 📁 Project Structure

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
⬆️ [Back to Top](#table-of-contents)

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

### Dependencies (`requirements.txt`)

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
⬆️ [Back to Top](#table-of-contents)

## 📦 Dataset

| Field | Value |
|---|---|
| Platform | YouTube (public) |
| Event | ICC Men's T20 World Cup |
| Video URL | https://www.youtube.com/watch?v=KvSriYbrGD8 |
| License | Public / Creative Commons |
| Duration used | 2 min 17 sec (3535 frames) |
| Resolution | 640 × 360 @ 30 FPS |

No custom dataset was collected. The pipeline uses **YOLOv8 pre-trained on COCO** (which includes the `person` class) applied directly to the cricket footage. No fine-tuning was required.

---
⬆️ [Back to Top](#table-of-contents)

## 🏋️ Training the Model

This project uses **pre-trained YOLOv8n weights** — no training from scratch was needed.

The model was applied **zero-shot** on cricket footage:

```
Model   : yolov8n.pt  (YOLOv8 Nano — COCO pre-trained)
Classes : person (class ID 0)
Conf    : >= 0.35
Device  : CPU / CUDA (auto-detected)
```

If you want to **fine-tune** on custom sports data:

```bash
yolo train \
  model=yolov8n.pt \
  data=your_dataset.yaml \
  epochs=50 \
  imgsz=640
```

---
⬆️ [Back to Top](#table-of-contents)

## 📈 Evaluation & Charts

All graphs are generated automatically from output CSVs using `graphs.py`:

```bash
python graphs.py --out-dir output/ --fps 30.0
```

### Graph 1 — Player Count per Frame
Shows how many players were visible frame-by-frame with a smoothed trend line.
The smoothed trend line removes single-frame noise to reveal crowd activity patterns.

![Player Count](https://raw.githubusercontent.com/anushabanoth-78/sports-tracker_project/main/output/graph_player_count.png)

### Graph 2 — Speed Distribution (km/h)
Histogram of all player speed estimates across the full video.
Calculated using **pixel displacement between consecutive frames × calibration factor**.
Shows the distribution of walking, jogging, and sprinting speeds.

![Speed Distribution](https://raw.githubusercontent.com/anushabanoth-78/sports-tracker_project/main/output/graph_speed_distribution.png)

### Graph 3 — Speed Timeline
Speed over time per tracked player ID.
Identifies which players were most active and when sprint events occurred.

![Speed Timeline](https://raw.githubusercontent.com/anushabanoth-78/sports-tracker_project/main/output/graph_speed_timeline.png)

### Graph 4 — ID Lifetime
How many frames each track ID survived before being lost or re-assigned.
Long bars = stable tracking; short bars = brief detection or ID swap.

![ID Lifetime](https://raw.githubusercontent.com/anushabanoth-78/sports-tracker_project/main/output/graph_id_lifetime.png)

### Auto-Screenshot Contact Sheet

<div align="center">

![contact](https://raw.githubusercontent.com/anushabanoth-78/sports-tracker_project/main/output/contact_sheet_ByteTrack.jpg)

*117 auto-screenshots combined — every 3 seconds + on every new player ID*

</div>

### Player Position Heatmap

<div align="center">

![heatmap](https://raw.githubusercontent.com/anushabanoth-78/sports-tracker_project/main/screenshots/heatmap_ByteTrack.jpg)

</div>

> 📄 [View Full Summary Report](https://github.com/anushabanoth-78/sports-tracker_project/blob/main/output/summary_report.txt)

---
⬆️ [Back to Top](#table-of-contents)
## 🔍 How It Works

```
Input Video
    │
    ▼
YOLOv8 Detection (per frame)
    │   → Detects all persons with confidence >= 0.35
    ▼
ByteTrack (ID assignment)
    │   → Two-stage matching: high-conf first, then low-conf recovery
    │   → Kalman filter predicts position during occlusion
    ▼
Annotation Layer
    │   → Bounding boxes (unique colour per ID)
    │   → Speed label (km/h via pixel displacement)
    │   → Motion trails (last 35 frames, fading)
    │   → HUD overlay (FPS, frame count, active/total IDs)
    ▼
Output
    ├──► Annotated video (tracked_v2.mp4)
    ├──► Auto-screenshots (every 3s + new ID events)
    ├──► Heatmap (cumulative player positions)
    ├──► CSVs (player count + speed per frame)
    └──► graphs.py → 4 analytics graphs + summary report

```
⬆️ [Back to Top](#table-of-contents)

### Tracker Comparison

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

ByteTrack was selected because it offered the best balance of speed and accuracy for fast-moving cricket footage. Its two-stage matching — associating high-confidence detections first, then recovering low-confidence ones — is particularly effective for motion-blurred frames during batting strokes. BoT-SORT showed 
slightly better re-identification after full occlusion, which is useful for dense player clusters.
⬆️ [Back to Top](#table-of-contents)

### Assumptions & Limitations

**Assumptions:**
- Input video is standard MP4 at reasonable resolution (480p or above)
- Players wear distinct jerseys — visually identical subjects may cause ID swaps
- Camera is roughly stationary or panning slowly — rapid zoom reduces accuracy
- Speed estimation uses a fixed pixel-per-metre calibration and is approximate

**Limitations:**
- **High ID count**: 65+ track IDs across a 2-minute video includes re-assignments due to occlusion and re-entry — not all are unique individuals
- **No ReID module**: A player who exits and re-enters the frame after >40 frames will receive a new ID
- **CPU speed**: ~8–10 FPS on CPU; a CUDA GPU provides 40–60 FPS
- **Motion blur**: Confidence drops to ~0.42 during high-speed batting strokes
- **Spectator false positives**: Background crowd near boundaries occasionally triggers detections

---
⬆️ [Back to Top](#table-of-contents)

## 🚀 Running the Tracker

```bash
python tracker.py \
    --source  public_cricket.mp4 \
    --output  output/tracked_output.mp4 \
    --model   yolov8n.pt \
    --tracker bytetrack.yaml
```
⬆️ [Back to Top](#table-of-contents)

### CLI Options

| Argument | Default | Description |
|---|---|---|
| `--source` | `public_cricket.mp4` | Input video path |
| `--output` | `output/tracked_output.mp4` | Annotated output video path |
| `--model` | `yolov8n.pt` | YOLOv8 weights (`n` / `m` / `l` / `x`) |
| `--tracker` | `bytetrack.yaml` | `bytetrack.yaml` or `botsort.yaml` |
| `--skip` | `1` | Draw annotations every N frames (inference always runs) |
| `--show` | off | Show live preview window while processing |
| `--screenshot-dir` | auto | Custom folder for auto-screenshots |

---
⬆️ [Back to Top](#table-of-contents)

## 🧩 Code Modules

| File | Purpose |
|---|---|
| `tracker.py` | Main pipeline — loads YOLOv8, runs ByteTrack, draws annotations, saves output |
| `graphs.py` | Reads output CSVs, generates 4 analytics graphs + summary report |
| `requirements.txt` | All Python dependencies |
| `yolov8n.pt` | Pre-trained YOLOv8 Nano weights (COCO) |

### Key functions in `tracker.py`

| Function | Description |
|---|---|
| `draw_trails()` | Draws fading motion trail for each track ID |
| `estimate_speed()` | Converts pixel displacement to km/h |
| `save_screenshot()` | Auto-saves frames every 3s or on new ID |
| `draw_hud()` | Renders FPS, frame count, active/total IDs |
| `build_contact_sheet()` | Combines all screenshots into one grid image |

---
⬆️ [Back to Top](#table-of-contents)

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| YOLOv8 (Ultralytics) | >= 8.0.0 | Object detection |
| ByteTrack | via supervision | Multi-object tracking |
| OpenCV | >= 4.8.0 | Video I/O and annotation |
| NumPy | >= 1.24.0 | Array operations |
| Pandas | >= 2.0.0 | CSV data handling |
| Matplotlib | >= 3.7.0 | Graph generation |
| Supervision | >= 0.18.0 | Tracker integration |

---
⬆️ [Back to Top](#table-of-contents)

## 🔧 Future Improvements

- **ReID embeddings** — integrate appearance features (OSNet) to maintain IDs across full exits and re-entries
- **Top-view projection** — homography transform to bird's-eye court view for tactical analysis
- **Team clustering** — k-means on jersey colour histograms to auto-assign team labels
- **Ball trajectory prediction** — Kalman filter tuned for ballistic motion
- **Evaluation metrics** — HOTA / MOTA / IDF1 against hand-labelled ground truth
- **GPU deployment** — ONNX + TensorRT export for real-time 60 FPS inference
- **Web dashboard** — Streamlit app to upload video and view analytics live

---

⬆️ [Back to Top](#table-of-contents)
## 👩‍💻 Project Author

<div align="center">

| Field | Value |
|---|---|
| **Assignment** | Multi-Object Detection and Persistent ID Tracking in Public Sports Footage |
| **Type** | AI / Computer Vision / Data Science |
| **Author** | Banoth Anusha |
| **Institute** | IIT Goa |
| **Model** | YOLOv8n |
| **Tracker** | ByteTrack |
| **Language** | Python 3.10+ |

<br>

🔗 [GitHub](https://github.com/anushabanoth-78) &nbsp;|&nbsp; 📧 banoth.anusha.22031@iitgoa.ac.in

<br>

*Built with YOLOv8 + ByteTrack — ICC T20 Cricket — Computer Vision Assignment*

</div>




