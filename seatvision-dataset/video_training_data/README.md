# Video training-data staging

Put the two consented `.mov` recordings in this directory. Raw videos,
extracted frames, detector output, and review pages in this directory are
intentionally ignored by Git.

The room has eight physical seats in both recordings. Once the files are here,
run the preparation tool from the `seatvision-dataset` directory, substituting
the exact filenames:

```bash
python3 scripts/prepare_video_training_data.py \
  --video "first_video.mov=1" \
  --video "second_video.mov=4" \
  --expected-seat-count 8 \
  --sample-fps 2 \
  --run-id object-claim-2fps
```

The number after `=` is optional. When known, it is the session-level
occupied-seat count supplied by the operator; it is not treated as a per-chair
ground-truth label. Omit it for a review-only prelabel run.

The command creates:

```text
video_training_data/
  prepared/<run-id>/<session>/frames/       # sampled still frames for annotation
  prepared/<run-id>/<session>/manifest.json # source/timestamp/session metadata
  prelabels/<run-id>/<session>/             # current YOLO proposals
  reviews/<run-id>/<session>/
    occupancy_review.jsonl         # one review proposal per frame
    review.html                    # visual review gallery
```

`occupancy_review.jsonl` is deliberately **not** a training annotation file.
The generic COCO model cannot reliably find all eight usable seat surfaces or
assign a person or a belonging to a specific chair. The default proposal labels
include `backpack`, `handbag`, `suitcase`, `bottle`, `cup`, `book`, `cell
phone`, and `laptop`; an object is proposed only when its center lies in an
estimated lower-chair region. Use the proposals to accelerate human review,
then create the final `seatable_surface`, `chair`, `person`, `belonging`, and
occupancy labels in CVAT. Do not use a session-wide count to blindly label
every chair in every frame.

Two frames per second preserves more distinct phone-camera perspectives than
the original one-FPS run. The default near-duplicate filter compares 64x64
grayscale frames and skips almost identical samples. Use a new `--run-id` for
every sampling experiment; it never overwrites an earlier review batch.
