# Operations and validation

## Health and performance qualification

Before making availability claims, record the following for each deployment:

- source frame rate, dropped frames, and timestamp age;
- detector latency and queue depth;
- state-event P50/P95 latency;
- GPU/RAM/thermal readings (`tegrastats`);
- camera disconnect/reconnect behavior;
- false-available rate from an independently labeled replay set.

Run `tegrastats` during a demo or soak test:

```bash
tegrastats --interval 1000
```

The app must downgrade affected seats to `Unknown`/`Occluded` when perception is
stale or camera health is lost. It should never manufacture `Available` status.

## Engine provenance

Never commit a TensorRT engine. It is hardware/runtime-specific. The build script
prints SHA-256 values for the ONNX and engine; store those alongside the model
version, JetPack release, TensorRT version, calibration revision, and taxonomy
version in deployment records.

## Calibration qualification

Do not fuse two camera identities or report metric depth until the camera rig is
rigidly mounted and calibration has passed reprojection/skew checks. The current
IMX219 cameras can still provide independent corroborating views beforehand.

For a calibrated rig, use an offline Charuco/stereo capture workflow or a
qualified automatic calibration service. Keep calibration files versioned and
invalidate them if either camera or mount moves.

## Data / model lifecycle

The shipping COCO model validates ingestion and dynamic-scene code only. Build a
consented, venue-specific dataset with chair/seat-pan masks, people, generic
belongings, occluders, and sit/stand/item action timestamps. Split train/test by
session and day, not neighboring frames. Deploy candidate models in shadow mode
and compare per-seat/event metrics before changing live decisions.
