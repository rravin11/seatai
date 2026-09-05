#!/usr/bin/env python3
"""Prepare review-first still-image batches from consented seat videos.

This script separates extracted frames, generic-model proposals, and human
ground truth. It never creates a final occupied/available label automatically.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError as exc:  # pragma: no cover - environment diagnostic
    raise SystemExit(
        "OpenCV is required for video ingestion. On this Jetson, use the system "
        "OpenCV package; do not install the PyPI opencv-python wheel."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "video_training_data"
DEFAULT_DETECTOR = PROJECT_ROOT / "build" / "seatvision-dataset"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "coco_yolov8n.yaml"
VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi"}
# These are proposal-only COCO labels. A human still decides whether the
# detected object is physically on a usable seat rather than on a table.
DEFAULT_CLAIM_OBJECT_LABELS = {"backpack", "handbag", "suitcase", "bottle", "cup", "book", "cell phone", "laptop"}


@dataclass(frozen=True)
class VideoSession:
    source: Path
    session_id: str
    expected_occupied_count: int | None


def parse_session(value: str, data_dir: Path) -> VideoSession:
    filename, separator, occupied_text = value.rpartition("=")
    if not separator:
        filename = value
        expected_occupied_count = None
    else:
        try:
            expected_occupied_count = int(occupied_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("EXPECTED_OCCUPIED must be an integer") from exc
        if expected_occupied_count < 0:
            raise argparse.ArgumentTypeError("EXPECTED_OCCUPIED must be non-negative")
    source = (data_dir / filename).resolve()
    try:
        source.relative_to(data_dir.resolve())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Video must be inside video_training_data") from exc
    if source.suffix.lower() not in VIDEO_EXTENSIONS:
        raise argparse.ArgumentTypeError(f"Unsupported video extension: {source.suffix}")
    session_id = re.sub(r"[^a-z0-9]+", "_", source.stem.lower()).strip("_")
    if not session_id:
        raise argparse.ArgumentTypeError("Video filename must contain letters or numbers")
    return VideoSession(source, session_id, expected_occupied_count)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample phone/MIPI video into still frames and create human-review prelabels."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Video staging directory")
    parser.add_argument(
        "--video",
        action="append",
        required=True,
        metavar="FILENAME[=EXPECTED_OCCUPIED]",
        help="Repeat per recording; an optional expected count is a session-level review check, not ground truth.",
    )
    parser.add_argument("--expected-seat-count", type=int, default=8, help="Known physical seat count (default: 8)")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Frames sampled per second (default: 2)")
    parser.add_argument("--max-frames", type=int, default=0, help="Pilot cap per video; 0 means no cap")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality from 1 to 100 (default: 95)")
    parser.add_argument(
        "--dedupe-threshold",
        type=float,
        default=3.0,
        help="Skip a sample when its 64x64 grayscale mean pixel difference from the last kept frame is below this value; 0 disables it.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional output namespace, e.g. object-claim-2fps; preserves earlier preparation runs.",
    )
    parser.add_argument(
        "--claim-object-label",
        action="append",
        default=[],
        metavar="LABEL",
        help="COCO label to consider for an object-on-seat proposal; repeatable. Defaults to common belongings.",
    )
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR, help="Built seatvision-dataset executable")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Detector config")
    parser.add_argument("--skip-prelabels", action="store_true", help="Only extract frames; skip current-model proposals")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print work without writing files")
    args = parser.parse_args()
    if args.expected_seat_count < 1:
        parser.error("--expected-seat-count must be positive")
    if args.sample_fps <= 0:
        parser.error("--sample-fps must be positive")
    if args.max_frames < 0:
        parser.error("--max-frames must be non-negative")
    if args.dedupe_threshold < 0:
        parser.error("--dedupe-threshold must be non-negative")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    args.data_dir = args.data_dir.resolve()
    try:
        args.sessions = [parse_session(item, args.data_dir) for item in args.video]
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    session_ids = [session.session_id for session in args.sessions]
    if len(set(session_ids)) != len(session_ids):
        parser.error("Video filenames resolve to duplicate session IDs; rename one recording")
    args.run_id = re.sub(r"[^a-z0-9]+", "_", args.run_id.lower()).strip("_")
    if args.run_id == "":
        args.run_id = None
    args.claim_object_labels = set(args.claim_object_label or DEFAULT_CLAIM_OBJECT_LABELS)
    for session in args.sessions:
        if session.expected_occupied_count is not None and session.expected_occupied_count > args.expected_seat_count:
            parser.error(f"{session.source.name}: expected occupied count exceeds expected seat count")
        if not session.source.is_file():
            parser.error(f"Video not found: {session.source}")
    if not args.skip_prelabels:
        if not args.detector.is_file():
            parser.error(f"Detector executable not found: {args.detector}. Build the project first.")
        if not args.config.is_file():
            parser.error(f"Detector config not found: {args.config}")
    return args


def require_new_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise RuntimeError(f"Refusing to overwrite non-empty output directory: {path}")
    else:
        path.mkdir(parents=True)


def frame_signature(image: Any) -> Any:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grayscale, (64, 64), interpolation=cv2.INTER_AREA)


def extract_frames(
    session: VideoSession,
    output: Path,
    sample_fps: float,
    max_frames: int,
    jpeg_quality: int,
    dedupe_threshold: float,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(session.source))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {session.source}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(source_fps) or source_fps <= 0:
        capture.release()
        raise RuntimeError(f"Video has no usable frame rate: {session.source}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_every = max(1, round(source_fps / sample_fps))
    frames_dir = output / "frames"
    frames_dir.mkdir()
    records: list[dict[str, Any]] = []
    source_index = 0
    selected = 0
    sampled_candidates = 0
    duplicate_frames_skipped = 0
    last_kept_signature = None
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if source_index % sample_every == 0:
                sampled_candidates += 1
                signature = frame_signature(image)
                if last_kept_signature is not None and dedupe_threshold > 0:
                    mean_difference = float(cv2.mean(cv2.absdiff(signature, last_kept_signature))[0])
                    if mean_difference < dedupe_threshold:
                        duplicate_frames_skipped += 1
                        source_index += 1
                        continue
                filename = f"frame_{selected:06d}_t{source_index / source_fps:09.3f}s.jpg"
                target = frames_dir / filename
                if not cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                    raise RuntimeError(f"Failed to write frame: {target}")
                records.append(
                    {
                        "frame_file": f"frames/{filename}",
                        "source_frame_index": source_index,
                        "timestamp_seconds": round(source_index / source_fps, 6),
                    }
                )
                last_kept_signature = signature
                selected += 1
                if max_frames and selected >= max_frames:
                    break
            source_index += 1
    finally:
        capture.release()
    if not records:
        raise RuntimeError(f"No frames could be decoded from: {session.source}")
    return {
        "schema_version": "seatvision-video-extraction/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_video": session.source.name,
        "session_id": session.session_id,
        "source_video_properties": {
            "fps": source_fps,
            "frame_count_reported": frame_count,
            "width": width,
            "height": height,
        },
        "sampling": {
            "requested_fps": sample_fps,
            "source_frame_stride": sample_every,
            "jpeg_quality": jpeg_quality,
            "near_duplicate_mean_difference_threshold": dedupe_threshold,
            "sampled_candidates": sampled_candidates,
            "duplicate_frames_skipped": duplicate_frames_skipped,
            "kept_frames": selected,
        },
        "frames": records,
    }


def run_prelabel_detector(detector: Path, config: Path, frames_dir: Path, output: Path) -> None:
    command = [str(detector), "--input", str(frames_dir), "--output", str(output), "--config", str(config)]
    print("Running current-model proposals:", " ".join(command))
    subprocess.run(command, check=True)


def rectangle_from_detection(detection: dict[str, Any]) -> tuple[float, float, float, float]:
    x, y, width, height = detection["bbox_xywh"]
    return float(x), float(y), float(width), float(height)


def person_association_score(person: dict[str, Any], chair: dict[str, Any]) -> float:
    """Explainable geometric proposal only; never a ground-truth decision."""
    person_x, person_y, person_w, person_h = rectangle_from_detection(person)
    chair_x, chair_y, chair_w, chair_h = rectangle_from_detection(chair)
    foot_x = person_x + person_w * 0.5
    foot_y = person_y + person_h * 0.78
    left, right = chair_x - chair_w * 0.20, chair_x + chair_w * 1.20
    top, bottom = chair_y + chair_h * 0.05, chair_y + chair_h * 1.10
    if not (left <= foot_x <= right and top <= foot_y <= bottom):
        return 0.0
    overlap_left, overlap_top = max(person_x, chair_x), max(person_y, chair_y)
    overlap_right, overlap_bottom = min(person_x + person_w, chair_x + chair_w), min(person_y + person_h, chair_y + chair_h)
    overlap = max(0.0, overlap_right - overlap_left) * max(0.0, overlap_bottom - overlap_top)
    normalization = max(1.0, min(person_w * person_h, chair_w * chair_h))
    return min(1.0, 0.55 + 0.45 * overlap / normalization)


def object_on_seat_score(candidate: dict[str, Any], chair: dict[str, Any]) -> float:
    """Scores whether an object center falls in an estimated chair-seat region.

    Chair boxes include their back/rest. This deliberately favors the lower
    half where a seat pan usually projects, and is only a review prioritizer.
    """
    object_x, object_y, object_w, object_h = rectangle_from_detection(candidate)
    chair_x, chair_y, chair_w, chair_h = rectangle_from_detection(chair)
    center_x = object_x + object_w * 0.5
    center_y = object_y + object_h * 0.5
    support_left, support_right = chair_x - chair_w * 0.12, chair_x + chair_w * 1.12
    support_top, support_bottom = chair_y + chair_h * 0.42, chair_y + chair_h * 0.95
    if not (support_left <= center_x <= support_right and support_top <= center_y <= support_bottom):
        return 0.0
    relative_size = min(1.0, (object_w * object_h) / max(1.0, chair_w * chair_h))
    return min(1.0, 0.60 + 0.40 * relative_size)


def create_review_records(
    predictions_path: Path,
    expected_seats: int,
    expected_occupied: int | None,
    claim_object_labels: set[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        prediction = json.loads(line)
        detections = prediction.get("detections", [])
        chairs = sorted(
            [item for item in detections if item.get("semantic_role") == "seat"],
            key=lambda item: (item["bbox_xywh"][0], item["bbox_xywh"][1]),
        )
        people = [item for item in detections if item.get("semantic_role") == "person"]
        candidate_objects = [item for item in detections if item.get("label") in claim_object_labels]
        proposed_occupied: list[dict[str, Any]] = []
        proposed_object_claims: list[dict[str, Any]] = []
        for chair_index, chair in enumerate(chairs):
            best_score = max((person_association_score(person, chair) for person in people), default=0.0)
            if best_score >= 0.55:
                proposed_occupied.append({"frame_local_chair_index": chair_index, "heuristic_score": round(best_score, 4)})
            object_scores = [(object_on_seat_score(candidate, chair), candidate) for candidate in candidate_objects]
            best_object_score, best_object = max(object_scores, default=(0.0, None), key=lambda item: item[0])
            if best_object is not None and best_object_score >= 0.60:
                proposed_object_claims.append(
                    {
                        "frame_local_chair_index": chair_index,
                        "object_label": best_object["label"],
                        "object_detector_score": best_object["score"],
                        "heuristic_score": round(best_object_score, 4),
                    }
                )
        records.append(
            {
                "schema_version": "seatvision-occupancy-review/v1",
                "source": prediction["source"],
                "annotated_preview": prediction.get("annotated"),
                "detector_observation": {
                    "detected_chair_candidates": len(chairs),
                    "detected_people": len(people),
                    "candidate_belonging_labels": sorted(claim_object_labels),
                    "detected_candidate_belongings": len(candidate_objects),
                    "heuristic_person_occupied_chair_candidates": proposed_occupied,
                    "heuristic_object_claimed_chair_candidates": proposed_object_claims,
                    "heuristic_nonempty_chair_count": len({
                        item["frame_local_chair_index"] for item in proposed_occupied + proposed_object_claims
                    }),
                },
                "known_session_context": {
                    "physical_seat_count": expected_seats,
                    "expected_occupied_count": expected_occupied,
                    "scope": "human-supplied session-level check only",
                },
                "human_review": {"required": True, "status": "unreviewed", "final_individual_seat_labels": None},
                "warning": (
                    "This is a generic COCO detector plus geometry proposal. It is not a training label and may miss "
                    "chairs, usable seat surfaces, people, or occlusions."
                ),
            }
        )
    return records


def write_review_html(path: Path, records: list[dict[str, Any]], prelabels_dir: Path, review_dir: Path) -> None:
    relative_prelabels = Path(os.path.relpath(prelabels_dir, review_dir))
    cards: list[str] = []
    for record in records:
        preview = record["annotated_preview"]
        image = ""
        if preview:
            image_path = (relative_prelabels / preview).as_posix()
            image = f'<a href="{html.escape(image_path)}"><img loading="lazy" src="{html.escape(image_path)}"></a>'
        observation = record["detector_observation"]
        context = record["known_session_context"]
        cards.append(
            "<article><strong>" + html.escape(record["source"]) + "</strong><br>"
            + f"<small>Detector: {observation['detected_chair_candidates']} chair candidates, "
            + f"{observation['detected_people']} people, {observation['detected_candidate_belongings']} candidate belongings, "
            + f"{observation['heuristic_nonempty_chair_count']} non-empty-seat proposals. "
            + f"Session check: {context['expected_occupied_count'] if context['expected_occupied_count'] is not None else 'not set'} "
            + f"occupied of {context['physical_seat_count']} seats.</small>"
            + image
            + "<p>Review required — do not treat this proposal as ground truth.</p></article>"
        )
    path.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>SeatVision occupancy review</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;background:#101418;color:#e7edf3}"
        ".notice{padding:1rem;background:#3d2d20;border-left:4px solid #ffb35c}.grid{display:grid;"
        "grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1rem}article{background:#192129;padding:.75rem;"
        "border-radius:.4rem}img{display:block;width:100%;margin-top:.6rem;background:#000}</style></head><body>"
        "<h1>Occupancy proposal review</h1><div class=\"notice\"><strong>Not ground truth.</strong> "
        "Correct seats, people, belongings, seat surfaces, occlusion, and occupancy in a labeling tool before training."
        "</div><div class=\"grid\">"
        + "".join(cards)
        + "</div></body></html>\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_arguments()
    print(f"Using video staging directory: {args.data_dir}")
    for session in args.sessions:
        print(
            f"  {session.source.name}: session={session.session_id}, "
            f"known session occupancy={session.expected_occupied_count if session.expected_occupied_count is not None else 'not set'}"
            f"/{args.expected_seat_count}"
        )
    if args.dry_run:
        return 0
    for session in args.sessions:
        prepared_root = args.data_dir / "prepared"
        prelabels_root = args.data_dir / "prelabels"
        reviews_root = args.data_dir / "reviews"
        if args.run_id is not None:
            prepared_root /= args.run_id
            prelabels_root /= args.run_id
            reviews_root /= args.run_id
        prepared_dir = prepared_root / session.session_id
        prelabels_dir = prelabels_root / session.session_id
        review_dir = reviews_root / session.session_id
        require_new_directory(prepared_dir)
        try:
            manifest = extract_frames(
                session, prepared_dir, args.sample_fps, args.max_frames, args.jpeg_quality, args.dedupe_threshold
            )
            manifest["session_expectations"] = {
                "physical_seat_count": args.expected_seat_count,
                "expected_occupied_count": session.expected_occupied_count,
                "scope": "session-level human supplied review check; not a per-frame or per-seat annotation",
            }
            (prepared_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(f"Extracted {len(manifest['frames'])} frames: {prepared_dir / 'frames'}")
            if args.skip_prelabels:
                continue
            require_new_directory(prelabels_dir)
            run_prelabel_detector(args.detector.resolve(), args.config.resolve(), prepared_dir / "frames", prelabels_dir)
            records = create_review_records(
                prelabels_dir / "predictions.jsonl",
                args.expected_seat_count,
                session.expected_occupied_count,
                args.claim_object_labels,
            )
            require_new_directory(review_dir)
            with (review_dir / "occupancy_review.jsonl").open("w", encoding="utf-8") as output:
                for record in records:
                    output.write(json.dumps(record, sort_keys=True) + "\n")
            write_review_html(review_dir / "review.html", records, prelabels_dir, review_dir)
            print(f"Review {len(records)} proposals: {review_dir / 'review.html'}")
        except Exception:
            print(f"Failed while preparing {session.source.name}. Partial output was retained for diagnosis.", file=sys.stderr)
            raise
    print("Preparation complete. Review proposals, then create human-verified annotations before training.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Video training-data preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
