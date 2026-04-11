<<<<<<< HEAD
# Multi-Object Detection and Persistent ID Tracking in Public Sports Footage

> **YOLOv8 + ByteTrack / BoT-SORT pipeline** for real-time multi-person and ball tracking
> with motion trails, movement heatmaps, and frame-level analytics.

---

## Table of Contents

- [Overview](#overview)
- [Demo](#demo)
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

## Overview

This project implements an end-to-end computer vision pipeline that:

1. **Detects** players and sports balls in every frame using YOLOv8
2. **Assigns persistent IDs** to each detected subject using ByteTrack or BoT-SORT
3. **Visualises** bounding boxes, ID labels, and motion trails on an annotated output video
4. **Generates analytics** — heatmaps, object-count graphs, and a summary CSV

The pipeline is designed to remain stable across common real-world challenges:
occlusion, rapid movement, camera panning, and visually similar-looking players.

---

## Demo

| Input frame | Annotated output |
|---|---|
| *(raw video frame)* | *(bounding boxes + IDs + trails)* |

> 📹 See `demo/` for sample screenshots and the walkthrough video.

---

## Features

| Feature | Details |
|---|---|
| **Object detection** | YOLOv8n/m/l — COCO pre-trained, classes: person + sports ball |
| **Multi-object tracking** | ByteTrack (default) or BoT-SORT |
| **Persistent IDs** | Stable across occlusion and re-entry |
| **Motion trails** | Fading coloured path per track ID |
| **Movement heatmap** | Cumulative presence density saved as JPEG |
| **Object count CSV** | Frame-by-frame detection counts |
| **Analytics graphs** | 4 publication-ready graphs (see below) |
| **Auto-screenshots** | Saved at configurable frame numbers |
| **HUD overlay** | Live FPS, frame index, active count, total IDs |
| **Frame skipping** | Optional — speeds up processing on long videos |

---

## Pipeline Architecture

```
=======
Multi-Object Detection and Persistent ID Tracking in Public Sports Footage

YOLOv8 + ByteTrack / BoT-SORT pipeline for real-time multi-person and ball tracking
with motion trails, movement heatmaps, and frame-level analytics.


Table of Contents

Overview
Demo
Features
Pipeline Architecture
Installation
Usage
Output Examples
Analytics & Graphs
Tracker Comparison
Assumptions & Limitations
Possible Improvements
File Structure
Video Source


Overview
This project implements an end-to-end computer vision pipeline that:

Detects players and sports balls in every frame using YOLOv8
Assigns persistent IDs to each detected subject using ByteTrack or BoT-SORT
Visualises bounding boxes, ID labels, and motion trails on an annotated output video
Generates analytics — heatmaps, object-count graphs, and a summary CSV

The pipeline is designed to remain stable across common real-world challenges:
occlusion, rapid movement, camera panning, and visually similar-looking players.

Demo
Input frameAnnotated output(raw video frame)(bounding boxes + IDs + trails)

📹 See demo/ for sample screenshots and the walkthrough video.


Features
FeatureDetailsObject detectionYOLOv8n/m/l — COCO pre-trained, classes: person + sports ballMulti-object trackingByteTrack (default) or BoT-SORTPersistent IDsStable across occlusion and re-entryMotion trailsFading coloured path per track IDMovement heatmapCumulative presence density saved as JPEGObject count CSVFrame-by-frame detection countsAnalytics graphs4 publication-ready graphs (see below)Auto-screenshotsSaved at configurable frame numbersHUD overlayLive FPS, frame index, active count, total IDsFrame skippingOptional — speeds up processing on long videos

Pipeline Architecture
>>>>>>> b7e50d3 (Enhanced tracking visualization with speed, ball detection, and trajectory analysis)
Input video
    │
    ▼
┌─────────────────────┐
│  YOLOv8 Detector    │  ← detects persons (conf ≥ 0.45) and balls (conf ≥ 0.30)
│  (ultralytics)      │
└────────┬────────────┘
         │  raw detections (boxes, confidences, class IDs)
         ▼
┌─────────────────────┐
│  ByteTrack / BoT-SORT│  ← associates detections across frames → track IDs
│  (tracker config)   │
└────────┬────────────┘
         │  tracked objects (ID + box per frame)
         ▼
┌─────────────────────┐
│  Visualisation      │  ← bounding boxes, labels, motion trails, HUD
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Analytics          │  ← heatmap, CSV, graphs, screenshots
└─────────────────────┘
         │
         ▼
    Annotated output video  +  graphs/  +  screenshots/
<<<<<<< HEAD
```

---

## Installation

### Requirements

- Python 3.9+
- CUDA GPU recommended (CPU works but is slower)

### Steps

```bash
# 1. Clone the repository
=======

Installation
Requirements

Python 3.9+
CUDA GPU recommended (CPU works but is slower)

Steps
bash# 1. Clone the repository
>>>>>>> b7e50d3 (Enhanced tracking visualization with speed, ball detection, and trajectory analysis)
git clone https://github.com/<your-username>/sports-tracker.git
cd sports-tracker

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
<<<<<<< HEAD
```

### requirements.txt

```
=======
requirements.txt
>>>>>>> b7e50d3 (Enhanced tracking visualization with speed, ball detection, and trajectory analysis)
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
<<<<<<< HEAD
```

YOLOv8 weights (`yolov8n.pt`) are downloaded automatically on first run.

---

## Usage

### Run the tracker

```bash
python tracker.py \
=======
YOLOv8 weights (yolov8n.pt) are downloaded automatically on first run.

Usage
Run the tracker
bashpython tracker.py \
>>>>>>> b7e50d3 (Enhanced tracking visualization with speed, ball detection, and trajectory analysis)
    --source  public_cricket.mp4 \
    --output  output/tracked_output.mp4 \
    --model   yolov8n.pt \
    --tracker bytetrack.yaml
<<<<<<< HEAD
```

### All CLI options

| Argument | Default | Description |
|---|---|---|
| `--source` | `public_cricket.mp4` | Input video path |
| `--output` | `output/tracked_output.mp4` | Annotated output video |
| `--model` | `yolov8n.pt` | YOLOv8 weights (n / s / m / l / x) |
| `--tracker` | `bytetrack.yaml` | `bytetrack.yaml` or `botsort.yaml` |
| `--skip` | `1` | Process every N-th frame |
| `--show` | off | Show live preview window |
| `--heatmap` | off | Overlay heatmap on output video |
| `--no-csv` | off | Disable CSV export |
| `--shots-dir` | `screenshots` | Folder for auto-screenshots |

### Generate analytics graphs

```bash
python generate_graphs.py \
    --csv output/count_over_time.csv \
    --out graphs/
```

Produces four graphs in `graphs/`:

1. `graph1_frame_vs_count.png` — objects detected per frame + trend line
2. `graph2_count_distribution.png` — histogram of detection counts
3. `graph3_stats_summary.png` — peak / average / median / min bar chart
4. `graph4_activity_density.png` — activity density heatmap across time

---

## Output Examples

### Motion-trail annotated frame

```
[ screenshot: frame_00030.jpg ]
```

### Movement heatmap

```
[ screenshots/heatmap.jpg ]
```

---

## Analytics & Graphs

### Graph 1 — Objects per frame

Shows how many subjects were visible frame-by-frame.
The smoothed trend line removes single-frame noise to reveal crowd activity patterns.

### Graph 2 — Distribution

Histogram showing how often the scene contained 0, 1, 2 … N objects.
A tall spike at a certain count suggests a consistent formation or play pattern.

### Graph 3 — Statistics summary

Horizontal bar comparing peak, average, median, and minimum detections.
Quick at-a-glance summary for any reviewer.

### Graph 4 — Activity density map

Divides the video into time segments and colour-encodes average activity.
Bright segments indicate phases of high player presence; dark segments are calmer moments.

---

## Tracker Comparison

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

=======
All CLI options
ArgumentDefaultDescription--sourcepublic_cricket.mp4Input video path--outputoutput/tracked_output.mp4Annotated output video--modelyolov8n.ptYOLOv8 weights (n / s / m / l / x)--trackerbytetrack.yamlbytetrack.yaml or botsort.yaml--skip1Process every N-th frame--showoffShow live preview window--heatmapoffOverlay heatmap on output video--no-csvoffDisable CSV export--shots-dirscreenshotsFolder for auto-screenshots
Generate analytics graphs
bashpython generate_graphs.py \
    --csv output/count_over_time.csv \
    --out graphs/
Produces four graphs in graphs/:

graph1_frame_vs_count.png — objects detected per frame + trend line
graph2_count_distribution.png — histogram of detection counts
graph3_stats_summary.png — peak / average / median / min bar chart
graph4_activity_density.png — activity density heatmap across time


Output Examples
Motion-trail annotated frame
[ screenshot: frame_00030.jpg ]
Movement heatmap
[ screenshots/heatmap.jpg ]

Analytics & Graphs
Graph 1 — Objects per frame
Shows how many subjects were visible frame-by-frame.
The smoothed trend line removes single-frame noise to reveal crowd activity patterns.
Graph 2 — Distribution
Histogram showing how often the scene contained 0, 1, 2 … N objects.
A tall spike at a certain count suggests a consistent formation or play pattern.
Graph 3 — Statistics summary
Horizontal bar comparing peak, average, median, and minimum detections.
Quick at-a-glance summary for any reviewer.
Graph 4 — Activity density map
Divides the video into time segments and colour-encodes average activity.
Bright segments indicate phases of high player presence; dark segments are calmer moments.

Tracker Comparison
Both trackers were tested on the same video clip.
CriterionByteTrackBoT-SORTID stability on occlusionGoodVery goodID stability on re-entryModerateBetterProcessing speedFasterSlightly slowerMemory usageLowerHigherPerformance on fast motionGoodGoodImplementation complexitySimpleModerate
Chosen default: ByteTrack
>>>>>>> b7e50d3 (Enhanced tracking visualization with speed, ball detection, and trajectory analysis)
ByteTrack was selected because it offered the best balance of speed and accuracy
for fast-moving cricket footage where rapid ball movement is common.
BoT-SORT showed slightly better re-identification after full occlusion,
which is useful for team sports with dense player clusters.

<<<<<<< HEAD
---

## Assumptions & Limitations

**Assumptions**

- Input video is standard MP4 / AVI at a reasonable resolution (720p or above).
- Players wear distinct jerseys — visually identical subjects may cause ID swaps.
- Camera is roughly stationary or panning slowly. Rapid zoom can reduce accuracy.

**Limitations**

- The ball (COCO class 32) is detected only when clearly visible and above size threshold.
- In very crowded scenes (10+ overlapping players), short-term ID swaps can occur.
- `yolov8n` (nano) is used by default for speed; switching to `yolov8m` noticeably improves
  detection recall at the cost of roughly 2× processing time.
- No re-identification module is used — a player who exits and re-enters the frame
  will receive a new ID.

---

## Possible Improvements

1. **Re-identification module** — integrate appearance embeddings (e.g. OSNet) to
   maintain IDs across full exits and re-entries.
2. **Top-view projection** — homography transform to bird's-eye court/field view
   for tactical positional analysis.
3. **Speed estimation** — calibrate pixels-per-metre and compute per-player speed
   in km/h using the trajectory data already collected.
4. **Team clustering** — k-means on jersey colour histograms to auto-assign team labels.
5. **Ball trajectory prediction** — Kalman filter tuned for ballistic motion to
   hold the ball ID through brief occlusions.
6. **Evaluation metrics** — implement HOTA / MOTA / IDF1 against a hand-labelled
   ground truth for a short clip.

---

## File Structure

```
=======
Assumptions & Limitations
Assumptions

Input video is standard MP4 / AVI at a reasonable resolution (720p or above).
Players wear distinct jerseys — visually identical subjects may cause ID swaps.
Camera is roughly stationary or panning slowly. Rapid zoom can reduce accuracy.

Limitations

The ball (COCO class 32) is detected only when clearly visible and above size threshold.
In very crowded scenes (10+ overlapping players), short-term ID swaps can occur.
yolov8n (nano) is used by default for speed; switching to yolov8m noticeably improves
detection recall at the cost of roughly 2× processing time.
No re-identification module is used — a player who exits and re-enters the frame
will receive a new ID.


Possible Improvements

Re-identification module — integrate appearance embeddings (e.g. OSNet) to
maintain IDs across full exits and re-entries.
Top-view projection — homography transform to bird's-eye court/field view
for tactical positional analysis.
Speed estimation — calibrate pixels-per-metre and compute per-player speed
in km/h using the trajectory data already collected.
Team clustering — k-means on jersey colour histograms to auto-assign team labels.
Ball trajectory prediction — Kalman filter tuned for ballistic motion to
hold the ball ID through brief occlusions.
Evaluation metrics — implement HOTA / MOTA / IDF1 against a hand-labelled
ground truth for a short clip.


File Structure
>>>>>>> b7e50d3 (Enhanced tracking visualization with speed, ball detection, and trajectory analysis)
sports-tracker/
├── tracker.py              # Main tracking pipeline
├── generate_graphs.py      # Analytics graph generator
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── report/
│   └── technical_report.md # Short technical report (1–2 pages)
├── output/
│   ├── tracked_output.mp4  # Annotated output video
│   └── count_over_time.csv # Frame-level detection counts
├── graphs/
│   ├── graph1_frame_vs_count.png
│   ├── graph2_count_distribution.png
│   ├── graph3_stats_summary.png
│   └── graph4_activity_density.png
└── screenshots/
    ├── frame_00030.jpg
    ├── frame_00090.jpg
    ├── frame_00150.jpg
    ├── frame_00250.jpg
    └── heatmap.jpg
<<<<<<< HEAD
```

---

## Video Source

| Field | Value |
|---|---|
| Platform | YouTube (public) |
| Video title | *(add your video title here)* |
| URL | *(add your video URL here)* |
| License | Public / Creative Commons |
| Duration used | Full video / first N minutes |

---

*Assignment: Multi-Object Detection and Persistent ID Tracking in Public Sports Footage*
*Author: Anusha*
=======

Video Source
FieldValuePlatformYouTube (public)Video title(add your video title here)URL(add your video URL here)LicensePublic / Creative CommonsDuration usedFull video / first N minutes

Assignment: Multi-Object Detection and Persistent ID Tracking in Public Sports Footage
Author: Anusha
>>>>>>> b7e50d3 (Enhanced tracking visualization with speed, ball detection, and trajectory analysis)
