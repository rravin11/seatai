# SeatVision

SeatVision is a Jetson-native, dual-camera scene-understanding product
foundation for real-time seating intelligence. It intentionally does **not**
contain fixed seat polygons, chair IDs, or a list of belongings in C++ code.

```text
IMX219 / Argus capture → timestamped camera streams → TensorRT inference
  → semantic tracking → dynamic seat discovery → temporal state reasoning
  → privacy-preserving seat events
```

`seatvisiond` is the product executable. The earlier `seatvision` target is
kept only as a simple visual demo while the product path matures.

## Run on this Jetson

```bash
cd /home/rahulravindranath/projects/seatai/seatvision
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j"$(nproc)"
./scripts/download_yolo_model.sh
./scripts/build_tensorrt_engine.sh
./build/seatvisiond --config config/jetson.yaml
```

Press `q` or `Esc` to stop the viewer. The process writes compact seat
transition events to `runtime/seat-events.jsonl`; it does not record video.

For logic-only validation:

```bash
ctest --test-dir build --output-on-failure
```

### Camera-capture diagnostic

`seatvision_capture_probe` isolates the Jetson Argus → native GStreamer
`appsink` bridge. It does not load TensorRT, OpenCV video capture, tracking, or
the UI. Run it after changing MIPI hardware, sensor overlays, or the capture
pipeline:

```bash
./build/seatvision_capture_probe 0 60
./build/seatvision_capture_probe 1 60
```

Each command must finish with `PASS` and report BGRx 1280×720 caps. The
product uses this same native capture pipeline; it does not use
`cv::VideoCapture` for the MIPI cameras.

### Camera orientation

Camera orientation belongs in `config/jetson.yaml`, not in the display layer.
The installed IMX219 cameras use `rotate_180: 1`, which applies NVIDIA's
hardware `nvvidconv` 180° flip before inference, tracking, event generation,
and rendering. Set it to `0` for a normally mounted camera; the setting is
independent per sensor.

## What is dynamic today

- Camera count and sensor settings come from configuration.
- Model labels and semantic roles come from the versioned taxonomy in
  `config/jetson.yaml`.
- Persistent `chair`/`bench`/`couch` tracks become dynamic seat entities.
- Every non-person, non-seat semantic class is evaluated as a possible object
  on a discovered seat; the code does not special-case backpacks or bottles.
- Missing chair evidence becomes `Occluded` or `Unknown`, never an automatic
  `Available` result.

The generic COCO model is a plumbing and live-demo baseline. Its bounding boxes
cannot reliably identify a physical seating surface in crowded scenes. The
production model must be trained to segment `seatable_surface`, `chair`,
`human`, `belonging`, and `occluder` in this venue.

## Product design choices

- **TensorRT FP16** is the deployed inference path. Engines are built on this
  exact Orin and must be regenerated after model, TensorRT, CUDA, or JetPack
  changes.
- **No false availability:** causal timestamps, expiration windows, and
  transition confirmation prevent a delayed inference result from becoming
  current perception.
- **Multi-view honesty:** the IMX219 cameras are independent rolling-shutter
  sensors. Until their calibration and timing relationship are qualified,
  SeatVision treats them as independent views.
- **Model replaceability:** TensorRT seat-surface segmentation, pose, depth,
  and custom temporal behavior models can augment this pipeline without putting
  model labels into business logic.

See [the architecture document](docs/ARCHITECTURE.md) for contracts, safety
rules, and the calibrated multi-view/depth roadmap.

## Next model milestones

1. Train a compact seat-surface segmentation model on consented footage from
   these camera positions.
2. Add calibrated multi-view association and VPI depth ordering when the
   physical rig qualifies.
3. Run pose/action inference only on uncertain seat ROIs, not every frame.
4. Train a supervised temporal sit/stand/item-placement model from labeled
   sequences. Reinforcement learning is a later offline research concern, not
   a substitute for accurate perception labels.

## Licensing and privacy

The downloaded YOLO model is an evaluation dependency; confirm a model license
appropriate to your investor demo and commercial plans before distributing it.
Process locally, avoid identity recognition, retain events rather than video by
default, and collect explicit consent before using imagery for training.
