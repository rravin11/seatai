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
  --sample-fps 1
```

The number after `=` is the known session-level occupied-seat count supplied by
the operator. It is not treated as a per-chair ground-truth label.

The command creates:

```text
video_training_data/
  prepared/<session>/frames/       # sampled still frames for annotation
  prepared/<session>/manifest.json # source/timestamp/session metadata
  prelabels/<session>/             # current YOLO chair/person proposals
  reviews/<session>/
    occupancy_review.jsonl         # one review proposal per frame
    review.html                    # visual review gallery
```

`occupancy_review.jsonl` is deliberately **not** a training annotation file.
The generic COCO model cannot reliably find all eight usable seat surfaces or
assign a person to a specific chair. Use the proposals to accelerate human
review, then create the final `seatable_surface`, `chair`, `person`,
`belonging`, and occupancy labels in CVAT. Do not use a session-wide count to
blindly label every chair in every frame.
