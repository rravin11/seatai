# SeatAI changelog

## 2026-09-04 — Jetson camera integration and documented product baseline

### Added

- Separate CMake projects for live dual-camera perception (`seatvision`) and
  offline still-image validation (`seatvision-dataset`).
- Direct TensorRT YOLO adapter, dynamic semantic seat reasoning, tracking,
  timestamped event sink, dual-camera renderer, and logic tests in the live
  product.
- Offline recursive image ingestion, annotated output, prediction manifests,
  HTML reports, and optional COCO-style evaluation in the dataset lab.
- Native GStreamer `appsink` capture probe for isolated IMX219/Argus diagnosis.
- Local video-to-frame preparation workflow for consented phone/MIPI recordings
  with TensorRT chair/person proposals and explicit human-review manifests.
- Review-only video preparation for recordings whose known session occupancy
  count has not yet been matched to a specific filename.
- Object-on-seat review proposals for common belongings, plus named 2-FPS
  sampling runs with near-duplicate filtering.
- Local browser annotation GUI with the simplified human-reviewed box taxonomy:
  `chair`, `table`, and `object`; reviewed frames export as COCO JSON.

### Changed

- Replaced the live camera path's OpenCV `VideoCapture` bridge with native
  GStreamer appsink capture using negotiated `GstVideoFrame` buffers. This
  resolved the integrated empty-frame/NvBufSurface failure while raw GStreamer
  capture was healthy.
- Added per-camera `rotate_180` configuration and enabled it for both inverted
  IMX219 modules. Rotation is applied in the NVMM path before all inference
  and display consumers.
- Added Jetson environment guidance and project separation to the root README.

### Verified

- The real-time CMake build and scene-reasoner test pass.
- Both cameras pass independent native appsink probes at 1280x720 for 60
  frames, including concurrent execution.
- A dual-camera headless run reaches a first TensorRT inference result for each
  camera after the native capture fix.

### Known limits

- Generic COCO YOLO is not a venue-specific occupancy model and cannot provide
  a production availability guarantee.
- The present TensorRT engine emits a cross-device-plan warning and must be
  regenerated/validated before a production demo.
- Cameras are uncalibrated independent views; there is no depth fusion or
  cross-camera identity fusion.
