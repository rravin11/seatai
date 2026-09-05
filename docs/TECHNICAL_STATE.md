# SeatAI technical state

Last verified: 2026-09-04

## Deployment target

- Hardware: NVIDIA Jetson Orin Nano Developer Kit.
- Cameras: two Arducam IMX219 8 MP fixed-focus MIPI modules, exposed as Argus
  sensor IDs `0` and `1`.
- OS/runtime: L4T R39.2, Linux `6.8.12-1021-tegra`, CUDA 13.2, TensorRT
  10.16.2, GStreamer 1.24.2, CMake 3.28.3.
- Camera orientation: both modules are physically inverted. `rotate_180: 1`
  in `seatvision/config/jetson.yaml` applies `nvvidconv flip-method=2` before
  inference, tracking, event generation, and rendering. It is a per-camera
  setting, not a display-only transform.

## Repository layout

| Path | Purpose | Runtime boundary |
| --- | --- | --- |
| `seatvision/` | Real-time, dual-MIPI seating scene application. | C++20 / Argus / native GStreamer / TensorRT. |
| `seatvision-dataset/` | Offline still-image ingestion, annotation, and COCO evaluation lab. | C++20 / TensorRT; never opens a camera. |
| Root Python sources | Earlier desktop prototype using YOLO and monocular-depth experiments. | Not the Jetson production path. |

The live and offline projects are intentionally separate executables, build
trees, configurations, outputs, and responsibilities. They can share an ONNX
model or a TensorRT plan only when that plan is valid for the target runtime.

## Live application

`seatvisiond` is the live product executable. Its current path is:

```text
IMX219 / nvarguscamerasrc
  -> NVMM nvvidconv (+ optional 180 degree rotation)
  -> native GStreamer appsink / GstVideoFrame
  -> BGR OpenCV frame with timestamp
  -> bounded asynchronous TensorRT YOLO inference per camera
  -> role-based tracking
  -> dynamic chair/bench/couch entities
  -> temporal occupied / claimed / available / unknown reasoning
  -> renderer and privacy-preserving JSONL events
```

There are no source-coded seat polygons, chair identifiers, or specific
belonging labels in the live reasoning code. Class roles are configuration data
in `seatvision/config/jetson.yaml`.

### Camera-capture incident and resolution

The initial live implementation used OpenCV `cv::VideoCapture` to consume the
Argus pipeline. Raw `gst-launch` capture succeeded, but the integrated process
failed with `NvBufSurfaceFromFd Failed` and repeated empty reads. A native
GStreamer `appsink` probe isolated the issue from TensorRT and tracking.

The production capture path now uses the GStreamer C API directly, negotiates
the actual BGRx caps, maps frames through `GstVideoFrame` using the negotiated
row stride, and converts them to BGR only after mapping. The diagnostic target
is `seatvision_capture_probe`:

```bash
cd seatvision
./build/seatvision_capture_probe 0 60
./build/seatvision_capture_probe 1 60
```

Both probes passed concurrently on 2026-09-04: 60 mapped frames each at
1280x720 BGRx. The subsequent full dual-camera headless run produced a first
TensorRT result for each camera with no sustained capture-health fault.

## Model and product truth

The current live model is generic YOLOv8n trained on COCO classes. It is a
plumbing/demo baseline, not a deployed occupancy model:

- it can recognize generic `person`, `chair`, `bench`, `couch`, and common
  belongings such as `backpack` and `bottle`;
- it does not identify a usable seat pan, arbitrary desks/counters, or every
  chair under occlusion;
- `dining table` is the closest relevant COCO class; COCO has no general
  `desk` class;
- the real-time seat states are therefore demonstration evidence, not an
  accuracy or availability guarantee.

The TensorRT engine presently loads and runs, but TensorRT emits a
cross-device-plan warning. Before investor deployment or any JetPack,
TensorRT, CUDA, or model change, regenerate and validate the engine on this
specific Orin using `seatvision/scripts/build_tensorrt_engine.sh`; do not commit
the generated plan.

The next model-quality work is venue-specific supervised training for
`seatable_surface`, `chair`, `person`, `belonging`, and `occluder`, followed by
per-seat event evaluation. Calibrated multi-camera fusion, depth, and temporal
sit/stand models remain planned enhancements rather than current claims.

## Offline still-image lab

`seatvision-dataset` recursively processes a folder one image at a time and
writes annotated images, predictions, a manifest, JSON summaries, HTML report,
and optional COCO metrics. It never changes the real-time camera system.

```bash
cd seatvision-dataset
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j"$(nproc)"
./build/seatvision-dataset --input datasets/inbox --output runs/run-name
```

Images can be placed in `seatvision-dataset/datasets/inbox/`; images, model
plans, reports, events, and build artifacts are intentionally ignored by Git.

## Verification baseline

The following checks passed on 2026-09-04:

```bash
cd seatvision
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
./build/seatvision_capture_probe 0 60
./build/seatvision_capture_probe 1 60
timeout -s INT 14s ./build/seatvisiond --config config/jetson.yaml --headless
```

The test suite covers the temporal scene-reasoner logic. It does not prove
model accuracy, multi-camera calibration, or safety of an availability claim.
