# SeatVision Dataset Validation

This is a separate project from the sibling seatvision directory:

- seatvision is the real-time dual-MIPI-camera product.
- seatvision-dataset is an offline, still-image validation lab.

It uses the same TensorRT engine by default, but it has its own CMake build,
configuration, input datasets, output runs, reports, and tests. It never opens
a camera and never emits Available, Occupied, or other temporal seat-state
claims from a single photo.

## Build

~~~bash
cd /home/rahulravindranath/projects/seatai/seatvision-dataset
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
~~~

## Run a large image folder

Put or mount images anywhere outside runs, then point the tool at the folder.
It scans recursively and streams one image through the GPU at a time, so it
does not load the whole dataset into RAM.

~~~bash
./build/seatvision-dataset \
  --input /path/to/still-images \
  --output runs/baseline-001 \
  --config config/coco_yolov8n.yaml
~~~

Each run creates:

~~~text
runs/baseline-001/
  manifest.json       # engine/config provenance
  predictions.jsonl   # one raw prediction record per image
  errors.jsonl        # decode/inference/write issues, without aborting the batch
  summary.json        # label counts and mean/P50/P95 GPU inference time
  annotated/          # visual overlays, preserving source subdirectories
  report.html         # local visual review gallery
~~~

Open report.html locally to review detections. An unlabeled run proves that the
ingestion to TensorRT to output path works; it does not prove accuracy.

## Measure accuracy with COCO annotations

When images[].file_name in a standard COCO JSON file is relative to --input,
add:

~~~bash
./build/seatvision-dataset \
  --input /path/to/images \
  --output runs/labeled-001 \
  --ground-truth /path/to/instances.json
~~~

This adds metrics.json with per-class TP/FP/FN, precision, recall, F1,
AP@0.50, and AP@0.50:0.95. Categories outside the configured model taxonomy
are reported as explicit coverage gaps rather than silently scored.

Important baseline limitation: COCO has dining table, but no general desk or
table class. Do not map a desk to dining table unless a deliberately versioned
custom taxonomy/model makes that semantic equivalence valid. The current
baseline also detects furniture boxes, not seat surfaces or usable individual
seats.

--label-alias FROM=TO is available only for an intentional, auditable taxonomy
mapping; it never creates a model capability by itself.
