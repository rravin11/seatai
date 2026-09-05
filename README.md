# SeatAI

SeatAI is a computer-vision prototype for determining whether seats are available, occupied, claimed, or uncertain.

The current prototype combines:

1. **YOLO segmentation** for people and chairs
2. **Depth Anything V2** for monocular depth estimation
3. **Person-to-seat matching logic** using mask overlap, distance, and depth similarity

The next development milestone is live camera support with object tracking and temporal smoothing.

## Jetson projects

The production work is split into two independently buildable CMake projects:

- `seatvision/` is the real-time dual-MIPI Argus/TensorRT application (`seatvisiond`).
- `seatvision-dataset/` is the offline still-image lab (`seatvision-dataset`). It recursively ingests large image folders, writes annotated images and JSON reports, and optionally scores COCO annotations. It never opens cameras or emits temporal occupancy states.

Build the live application:

```bash
cd seatvision
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j"$(nproc)"
```

Build the still-image validator separately:

```bash
cd seatvision-dataset
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j"$(nproc)"
./build/seatvision-dataset --input /path/to/images --output runs/baseline-001
```

The default dataset config uses the same device-specific TensorRT engine as the
live app. See `seatvision-dataset/README.md` for COCO metrics, reports, and the
current taxonomy limitation: the generic model has `dining table`, not `desk`.

For the verified Jetson hardware/software state, capture diagnosis, current
model limits, and validation evidence, see [Technical state](docs/TECHNICAL_STATE.md).
All completed engineering slices follow the documented
[development workflow](docs/DEVELOPMENT_WORKFLOW.md) and are recorded in the
[changelog](docs/CHANGELOG.md).

## Current pipeline

```text
Camera or image
    ↓
YOLO segmentation
    ├── person masks
    └── chair masks
    ↓
Depth Anything V2
    └── per-pixel depth map
    ↓
Seat-candidate generation
    ↓
Person-to-seat matching
    ↓
available / occupied
```

The current pretrained YOLO model detects generic chairs and people. A future custom model should segment the usable seating surface and identify each seat more reliably.

## Project files

```text
seatai/
├── README.md
├── requirements.txt
├── .gitignore
├── seat_depth_segmentation.py
├── seat_camera_temporal.py
├── seat_occupancy.py
├── seat_state_engine.py
├── dynamic_seat_zones.py
├── seat_zones.py
├── models/
├── outputs/
├── tests/
└── seat_seg_dataset/
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

## Requirements

- Python 3.10 or 3.11 recommended
- A webcam, built-in Mac camera, or compatible external camera
- macOS, Windows, or Linux
- Internet access on the first run so model weights can download

On Apple Silicon, PyTorch can use the `mps` device. On NVIDIA systems, PyTorch can use CUDA.

### NVIDIA Jetson Orin (JetPack 7.2)

Use the Jetson-specific manifest on a Jetson running JetPack 7.2 (L4T R39.2,
Ubuntu 24.04, Python 3.12, CUDA 13). It reuses JetPack's OpenCV build, which
includes GStreamer and V4L2 camera support. Installing `opencv-python` from
PyPI instead can replace that build and break CSI-camera pipelines.

```bash
cd seatai
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-jetson.txt
```

Verify the CUDA runtime before running a model:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

It should print `True` for CUDA availability and identify the Orin GPU. First
model runs also need network access to download the YOLO and Depth Anything
weights. Camera availability is separate: connect the camera and check that a
`/dev/video*` device appears, or use the relevant GStreamer pipeline for a CSI
camera.

## Installation

```bash
git clone <repository-url>
cd seatai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running the image prototype

Update paths inside `seat_depth_segmentation.py`, then run:

```bash
python seat_depth_segmentation.py
```

Expected output:

```text
outputs/seat_depth_segmentation_output.png
outputs/seat_depth_map.png
```

## Connecting a camera

The next script should open a camera with:

```python
capture = cv2.VideoCapture(0)
```

An external camera may use index `1` or `2`.

The real-time pipeline should:

1. Read a frame.
2. Run YOLO segmentation.
3. Estimate depth.
4. Match people to seats.
5. Apply temporal smoothing.
6. Draw stabilized seat states.
7. Exit when `q` is pressed.

For speed, run depth every 3–10 frames and reuse the latest depth map between updates.

## Temporal smoothing

Do not change seat state from one frame alone.

Recommended initial behavior:

- confirm `occupied` after about 0.7–1.5 seconds
- confirm `available` after about 2–5 seconds
- preserve claimed-object evidence longer than one frame
- use persistent tracking IDs when available
- use an `uncertain` state when signals conflict

This reduces flickering caused by missed detections, motion blur, occlusion, and momentary depth errors.

## Suggested milestones

### Milestone 1: live camera
- Open the built-in camera with OpenCV
- Display segmentation and depth results
- Measure frames per second

### Milestone 2: temporal smoothing
- Add per-seat histories
- Add occupied/available hysteresis
- Add person tracking IDs

### Milestone 3: custom seat segmentation
- Collect images under varied lighting
- Label visible seat surfaces with polygons
- Train a custom YOLO segmentation model
- Replace generic chair masks with predicted seat masks

### Milestone 4: deployment
- Support multiple tables
- Add calibration
- Test external cameras
- Move inference to an NVIDIA Jetson
- Store seat states instead of raw video when possible

## Source control

Do not commit:

- `.venv/`
- model weights such as `*.pt`
- Python caches
- output images and videos
- training runs
- raw datasets
- IDE metadata

Do commit:

- Python scripts
- README files
- `requirements.txt`
- YAML configuration
- tests
- small dataset labels when practical

Large models and datasets should use cloud storage, release assets, or Git LFS.

## Cleaning thousands of Git changes

First back up or commit any source files you need.

If generated folders were already tracked, remove them from Git's index without deleting local copies:

```bash
git rm -r --cached .venv
git rm -r --cached outputs
git rm -r --cached runs
git rm -r --cached __pycache__
```

Some paths may report that they were not tracked. That is harmless.

To rebuild the index according to `.gitignore`:

```bash
git rm -r --cached .
git add .
git status
```

Review the list carefully, then commit:

```bash
git commit -m "Clean repository and add project setup files"
```

This does not delete ignored files from your computer. It only stops Git from tracking them.

## Collaboration workflow

```bash
git checkout -b feature/live-camera
git add seat_camera_temporal.py
git commit -m "Add live camera capture"
git push -u origin feature/live-camera
```

Avoid `git add .` until the `.gitignore` has been verified.

## Privacy

- avoid storing raw video unless necessary
- process locally when possible
- avoid face recognition
- document retention rules
- clearly communicate that occupancy sensing is in use
