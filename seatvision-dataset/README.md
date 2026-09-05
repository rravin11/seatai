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

## Prepare phone or MIPI recordings for labeling

Place consented recordings in `video_training_data/` and use the local video
preparation script. It samples still frames, runs the current TensorRT baseline
as a proposal generator, and creates a review gallery plus JSONL review
records. It does not turn a generic detector estimate into ground truth.

For the current two recordings—eight physical seats in both, with one and four
known occupied seats respectively—replace the filenames in:

~~~bash
python3 scripts/prepare_video_training_data.py \
  --video "first_video.mov=1" \
  --video "second_video.mov=4" \
  --expected-seat-count 8 \
  --sample-fps 2 \
  --run-id object-claim-2fps
~~~

The baseline can miss chairs, usable seat surfaces, people, belongings, and
occlusions. It creates separate person-on-chair and object-on-seat proposals;
the latter considers common COCO belongings but can still confuse an object on
a table with an object on a chair. `expected-seat-count` and supplied occupancy
counts are therefore session-level quality checks only. Review and correct each
selected frame in CVAT before making final segmentation/occupancy training
annotations. See `video_training_data/README.md` for output layout and
privacy/Git boundaries.
If the one-versus-four mapping is not yet known for a filename, omit `=COUNT`
for that video to create review-only proposals without inventing a label.

## Human annotation GUI: chair, table, object

Use the local browser GUI to turn proposals into human-reviewed box labels.
The deliberately small first taxonomy is exactly `chair`, `table`, and
`object`. Existing `chair`/`bench`/`couch` proposals become `chair`, `dining
table` becomes `table`, and every other generic proposal becomes `object` for
you to accept, relabel, delete, or redraw.

~~~bash
python3 scripts/label_training_data.py --open-browser
~~~

It opens `http://127.0.0.1:8765/` on the Jetson and saves an editable local
annotation file under `datasets/raw_videos/phone/human_annotations/`, which is
ignored by Git. Click a box to select it; use Chair/Table/Object to change its
class, drag empty image space to draw a new box, and Delete to remove a box.
Mark a frame reviewed only after checking its useful boxes, then use **Export
reviewed COCO**. The COCO JSON contains only reviewed images.

This is a box-level detector dataset, not a seat-surface segmentation dataset
or final occupancy ground truth. It gives us a simple, inspectable first model
baseline; later we will add seat-surface masks and object-to-seat relationships.
