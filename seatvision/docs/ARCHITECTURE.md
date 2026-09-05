# SeatVision product architecture

## Safety boundary

SeatVision never has source-coded seat coordinates, chair identifiers, or named
belonging types. A versioned model taxonomy defines semantic roles; runtime
tracking then creates and retires `DynamicSeat` entities from observed seatable
instances. The system returns `Unknown` or `Occluded` when evidence expires—it
does not infer `Available` from a missing camera/model observation.

## Runtime data flow

```text
Argus cameras → timestamped latest-frame streams → bounded async TensorRT jobs
  → raw instance observations → role-based per-view tracker
  → dynamic seat discovery / support-region geometry
  → temporal occupancy evidence → live renderer / event sink
```

The deployed fast path is C++/GStreamer/Argus/CUDA/TensorRT. Python and the
repository virtual environment are not runtime dependencies.

## Model contracts

`Detector` accepts an image and yields labels, scores, boxes, and optionally
masks. It has two adapters: the OpenCV ONNX adapter for debugging and a direct
TensorRT adapter for production. The semantic mapping is configuration rather
than C++ constants. Add segmentation, pose, depth, and open-vocabulary adapters
behind the same observation contract; do not put their label logic in the
occupancy engine.

## Multi-view/depth roadmap

The two IMX219 cameras are independent rolling-shutter cameras. They should be
treated as independent views until their mount is rigid, time skew is measured,
and calibration passes validation. Calibration then enables projection/fusion;
stereo depth is evidence for occlusion ordering, not an unquestioned source of
metric truth. A calibration revision must travel with every scene observation.

## Required accuracy work

A generic COCO model validates the product plumbing, but it is not a final seat
model. Train a venue-representative model with `seatable_surface`, `chair`,
`human`, `belonging`, and `occluder` masks; evaluate per-seat state/event F1,
false-available rate, transition latency, tracking continuity, and camera-fault
behavior. Use a supervised temporal action model before considering any offline
reinforcement-learning policy.
