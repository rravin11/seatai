from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = "/Users/rahulravindranath/Desktop/seatai/people_dining1.png"

OUTPUT_DIR = "/Users/rahulravindranath/Desktop/seatai/outputs"
OUTPUT_IMAGE_PATH = os.path.join(
    OUTPUT_DIR,
    "seat_depth_segmentation_output.png",
)
OUTPUT_DEPTH_PATH = os.path.join(
    OUTPUT_DIR,
    "seat_depth_map.png",
)

SEGMENTATION_MODEL_PATH = "yolo11m-seg.pt"

DEPTH_MODEL_ID = (
    "depth-anything/"
    "Depth-Anything-V2-Metric-Indoor-Small-hf"
)

YOLO_CONFIDENCE = 0.20
PERSON_CONFIDENCE = 0.50
CHAIR_CONFIDENCE = 0.25

MAX_SEATS = 4
MIN_MASK_PIXELS = 100

# Person-seat matching weights.
DEPTH_WEIGHT = 0.45
DISTANCE_WEIGHT = 0.35
OVERLAP_WEIGHT = 0.20

# A match below this score is rejected.
MATCH_THRESHOLD = 0.22

# Controls how quickly depth compatibility falls as depths differ.
DEPTH_LOG_SIGMA = 0.40


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Instance:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    center: tuple[float, float]
    median_depth: Optional[float] = None


@dataclass
class SeatCandidate:
    seat_id: str
    confidence: float
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    center: tuple[float, float]
    median_depth: float
    state: str = "available"
    assigned_person_index: Optional[int] = None
    match_score: float = 0.0


# ============================================================
# DEVICE SELECTION
# ============================================================

def select_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# DEPTH ESTIMATION
# ============================================================

def load_depth_model(
    device: torch.device,
) -> tuple[AutoImageProcessor, AutoModelForDepthEstimation]:
    processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)

    model = AutoModelForDepthEstimation.from_pretrained(
        DEPTH_MODEL_ID
    )
    model.to(device)
    model.eval()

    return processor, model


def estimate_depth(
    bgr_image: np.ndarray,
    processor: AutoImageProcessor,
    model: AutoModelForDepthEstimation,
    device: torch.device,
) -> np.ndarray:
    """
    Returns a floating-point depth map resized to the original image.

    For the selected indoor metric model, values are intended to represent
    estimated indoor metric depth. For assignment, this program mostly uses
    relative depth compatibility rather than trusting every value as an exact
    physical measurement.
    """
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    inputs = processor(
        images=pil_image,
        return_tensors="pt",
    )
    inputs = {
        name: tensor.to(device)
        for name, tensor in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    original_height, original_width = bgr_image.shape[:2]

    predicted_depth = F.interpolate(
        predicted_depth.unsqueeze(1),
        size=(original_height, original_width),
        mode="bicubic",
        align_corners=False,
    ).squeeze(1)

    return predicted_depth[0].detach().cpu().numpy().astype(
        np.float32
    )


def save_depth_visualization(
    depth_map: np.ndarray,
    output_path: str,
) -> None:
    finite_depth = depth_map[np.isfinite(depth_map)]

    if finite_depth.size == 0:
        raise ValueError("Depth model returned no finite depth values.")

    low = np.percentile(finite_depth, 2)
    high = np.percentile(finite_depth, 98)

    normalized = np.clip(
        (depth_map - low) / max(high - low, 1e-6),
        0.0,
        1.0,
    )

    depth_uint8 = (normalized * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(
        depth_uint8,
        cv2.COLORMAP_TURBO,
    )

    cv2.imwrite(output_path, depth_colormap)


# ============================================================
# SEGMENTATION
# ============================================================

def resize_binary_mask(
    mask: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    if mask.shape == (height, width):
        return mask.astype(bool)

    resized = cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    return resized.astype(bool)


def mask_center(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return 0.0, 0.0

    return float(np.mean(xs)), float(np.mean(ys))


def extract_segmented_instances(
    result,
    image_shape: tuple[int, int, int],
) -> list[Instance]:
    height, width = image_shape[:2]
    instances: list[Instance] = []

    if result.boxes is None or result.masks is None:
        return instances

    raw_masks = result.masks.data.detach().cpu().numpy()

    for index, box in enumerate(result.boxes):
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        label = result.names[class_id]

        if label == "person" and confidence < PERSON_CONFIDENCE:
            continue

        if label == "chair" and confidence < CHAIR_CONFIDENCE:
            continue

        if label not in {"person", "chair"}:
            continue

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist(),
        )

        mask = resize_binary_mask(
            raw_masks[index],
            width,
            height,
        )

        if int(mask.sum()) < MIN_MASK_PIXELS:
            continue

        instances.append(
            Instance(
                label=label,
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                mask=mask,
                center=mask_center(mask),
            )
        )

    return instances


# ============================================================
# MASK AND DEPTH UTILITIES
# ============================================================

def median_depth_in_mask(
    depth_map: np.ndarray,
    mask: np.ndarray,
) -> float:
    values = depth_map[
        mask
        & np.isfinite(depth_map)
        & (depth_map > 0)
    ]

    if values.size == 0:
        return float("nan")

    return float(np.median(values))


def lower_body_mask(person: Instance) -> np.ndarray:
    """
    Restricts a person mask to the lower portion of their bounding box.

    This is generally more relevant to chair assignment than the person's
    head and upper torso.
    """
    x1, y1, x2, y2 = person.bbox

    lower_start = int(y1 + 0.48 * (y2 - y1))

    restricted = person.mask.copy()
    restricted[:lower_start, :] = False

    if int(restricted.sum()) < MIN_MASK_PIXELS:
        return person.mask

    return restricted


def seat_contact_mask(chair: Instance) -> np.ndarray:
    """
    Produces an estimated chair seating/contact area from a generic chair mask.

    This is only a temporary heuristic. A custom-trained seat segmentation
    model should eventually replace it.
    """
    x1, y1, x2, y2 = chair.bbox

    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    region_x1 = int(x1 + 0.08 * width)
    region_x2 = int(x2 - 0.08 * width)
    region_y1 = int(y1 + 0.28 * height)
    region_y2 = int(y1 + 0.82 * height)

    region = np.zeros_like(chair.mask, dtype=bool)
    region[
        max(0, region_y1):max(0, region_y2),
        max(0, region_x1):max(0, region_x2),
    ] = True

    contact = chair.mask & region

    if int(contact.sum()) < MIN_MASK_PIXELS:
        return chair.mask

    return contact


def mask_overlap_score(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    denominator = max(1, min(mask_a.sum(), mask_b.sum()))

    return float(intersection / denominator)


def normalized_distance_score(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    image_shape: tuple[int, int, int],
) -> float:
    height, width = image_shape[:2]
    image_diagonal = math.hypot(width, height)

    distance = math.hypot(
        point_a[0] - point_b[0],
        point_a[1] - point_b[1],
    )

    return max(
        0.0,
        1.0 - distance / max(image_diagonal * 0.42, 1.0),
    )


def depth_compatibility_score(
    person_depth: float,
    seat_depth: float,
) -> float:
    if not np.isfinite(person_depth) or not np.isfinite(seat_depth):
        return 0.0

    if person_depth <= 0 or seat_depth <= 0:
        return 0.0

    log_ratio = abs(
        math.log(
            (person_depth + 1e-6)
            / (seat_depth + 1e-6)
        )
    )

    return math.exp(
        -0.5 * (log_ratio / DEPTH_LOG_SIGMA) ** 2
    )


# ============================================================
# AUTOMATIC SEAT CANDIDATES
# ============================================================

def assign_position_names(
    seats: list[SeatCandidate],
) -> None:
    """
    Temporary seat naming for a four-seat table.

    Two visually higher seats are called back-left/back-right.
    Two visually lower seats are called left-front/right-front.

    This naming logic assumes a similar perspective to the current test scene.
    A custom seat model should eventually predict seat identities directly.
    """
    if len(seats) != 4:
        for index, seat in enumerate(seats, start=1):
            seat.seat_id = f"seat_{index}"
        return

    ordered_by_y = sorted(
        seats,
        key=lambda seat: seat.center[1],
    )

    back = sorted(
        ordered_by_y[:2],
        key=lambda seat: seat.center[0],
    )
    front = sorted(
        ordered_by_y[2:],
        key=lambda seat: seat.center[0],
    )

    back[0].seat_id = "seat_back_left"
    back[1].seat_id = "seat_back_right"
    front[0].seat_id = "seat_left_front"
    front[1].seat_id = "seat_right_front"


def build_seat_candidates(
    chairs: list[Instance],
    depth_map: np.ndarray,
) -> list[SeatCandidate]:
    chairs = sorted(
        chairs,
        key=lambda chair: chair.confidence,
        reverse=True,
    )[:MAX_SEATS]

    seats: list[SeatCandidate] = []

    for chair in chairs:
        contact_mask = seat_contact_mask(chair)
        contact_center = mask_center(contact_mask)
        seat_depth = median_depth_in_mask(
            depth_map,
            contact_mask,
        )

        seats.append(
            SeatCandidate(
                seat_id="unassigned",
                confidence=chair.confidence,
                bbox=chair.bbox,
                mask=contact_mask,
                center=contact_center,
                median_depth=seat_depth,
            )
        )

    assign_position_names(seats)

    return seats


# ============================================================
# PERSON-TO-SEAT MATCHING
# ============================================================

def person_seat_match_score(
    person: Instance,
    person_mask: np.ndarray,
    person_depth: float,
    seat: SeatCandidate,
    image_shape: tuple[int, int, int],
) -> tuple[float, dict[str, float]]:
    depth_score = depth_compatibility_score(
        person_depth,
        seat.median_depth,
    )

    distance_score = normalized_distance_score(
        mask_center(person_mask),
        seat.center,
        image_shape,
    )

    overlap_score = mask_overlap_score(
        person_mask,
        seat.mask,
    )

    total = (
        DEPTH_WEIGHT * depth_score
        + DISTANCE_WEIGHT * distance_score
        + OVERLAP_WEIGHT * overlap_score
    )

    details = {
        "depth": depth_score,
        "distance": distance_score,
        "overlap": overlap_score,
    }

    return float(total), details


def assign_people_to_seats(
    people: list[Instance],
    seats: list[SeatCandidate],
    depth_map: np.ndarray,
    image_shape: tuple[int, int, int],
) -> list[dict]:
    candidates: list[dict] = []

    for person_index, person in enumerate(people):
        person_region = lower_body_mask(person)
        person_depth = median_depth_in_mask(
            depth_map,
            person_region,
        )

        person.median_depth = person_depth

        for seat_index, seat in enumerate(seats):
            score, details = person_seat_match_score(
                person,
                person_region,
                person_depth,
                seat,
                image_shape,
            )

            candidates.append(
                {
                    "person_index": person_index,
                    "seat_index": seat_index,
                    "score": score,
                    "details": details,
                    "person_mask": person_region,
                }
            )

    candidates.sort(
        key=lambda candidate: candidate["score"],
        reverse=True,
    )

    used_people: set[int] = set()
    used_seats: set[int] = set()
    assignments: list[dict] = []

    for candidate in candidates:
        person_index = candidate["person_index"]
        seat_index = candidate["seat_index"]

        if candidate["score"] < MATCH_THRESHOLD:
            continue

        if person_index in used_people:
            continue

        if seat_index in used_seats:
            continue

        seats[seat_index].state = "occupied"
        seats[seat_index].assigned_person_index = person_index
        seats[seat_index].match_score = candidate["score"]

        used_people.add(person_index)
        used_seats.add(seat_index)
        assignments.append(candidate)

    return assignments


# ============================================================
# DRAWING
# ============================================================

def mask_to_contours(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    return contours


def draw_mask_outline(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    contours = mask_to_contours(mask)

    cv2.drawContours(
        image,
        contours,
        contourIdx=-1,
        color=color,
        thickness=thickness,
    )


def draw_debug_output(
    image: np.ndarray,
    people: list[Instance],
    seats: list[SeatCandidate],
    assignments: list[dict],
) -> np.ndarray:
    output = image.copy()

    for person_index, person in enumerate(people):
        draw_mask_outline(
            output,
            person.mask,
            color=(255, 0, 255),
            thickness=2,
        )

        x1, y1, _, _ = person.bbox

        cv2.putText(
            output,
            (
                f"person_{person_index} "
                f"d={person.median_depth:.2f}"
            ),
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
        )

    for seat in seats:
        color = (
            (0, 0, 255)
            if seat.state == "occupied"
            else (0, 255, 0)
        )

        draw_mask_outline(
            output,
            seat.mask,
            color=color,
            thickness=3,
        )

        center = (
            int(seat.center[0]),
            int(seat.center[1]),
        )

        cv2.circle(
            output,
            center,
            6,
            color,
            -1,
        )

        x1, y1, _, _ = seat.bbox

        cv2.putText(
            output,
            (
                f"{seat.seat_id}: {seat.state} "
                f"d={seat.median_depth:.2f} "
                f"s={seat.match_score:.2f}"
            ),
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
        )

    for assignment in assignments:
        person = people[assignment["person_index"]]
        seat = seats[assignment["seat_index"]]

        person_center = mask_center(
            assignment["person_mask"]
        )

        cv2.line(
            output,
            (
                int(person_center[0]),
                int(person_center[1]),
            ),
            (
                int(seat.center[0]),
                int(seat.center[1]),
            ),
            (255, 255, 255),
            2,
        )

    return output


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {IMAGE_PATH}"
        )

    device = select_torch_device()
    print(f"Using Torch device: {device}")

    print("Loading depth model...")
    depth_processor, depth_model = load_depth_model(device)

    print("Estimating image depth...")
    depth_map = estimate_depth(
        image,
        depth_processor,
        depth_model,
        device,
    )

    save_depth_visualization(
        depth_map,
        OUTPUT_DEPTH_PATH,
    )

    print("Loading YOLO segmentation model...")
    segmentation_model = YOLO(SEGMENTATION_MODEL_PATH)

    results = segmentation_model.predict(
        source=IMAGE_PATH,
        conf=YOLO_CONFIDENCE,
        retina_masks=True,
        save=False,
        verbose=False,
    )

    if not results:
        raise RuntimeError("YOLO returned no result object.")

    instances = extract_segmented_instances(
        results[0],
        image.shape,
    )

    people = [
        instance
        for instance in instances
        if instance.label == "person"
    ]

    chairs = [
        instance
        for instance in instances
        if instance.label == "chair"
    ]

    print(f"Detected people: {len(people)}")
    print(f"Detected chairs: {len(chairs)}")

    if not chairs:
        print(
            "No chairs were segmented. Try lowering CHAIR_CONFIDENCE "
            "or use more suitable test imagery."
        )
        return

    seats = build_seat_candidates(
        chairs,
        depth_map,
    )

    assignments = assign_people_to_seats(
        people,
        seats,
        depth_map,
        image.shape,
    )

    print("\nSeat results:")

    for seat in seats:
        print(
            f"{seat.seat_id}: {seat.state} | "
            f"depth={seat.median_depth:.3f} | "
            f"match={seat.match_score:.3f}"
        )

    print("\nAccepted person-seat assignments:")

    for assignment in assignments:
        seat = seats[assignment["seat_index"]]
        details = assignment["details"]

        print(
            f"person_{assignment['person_index']} -> "
            f"{seat.seat_id} | "
            f"total={assignment['score']:.3f}, "
            f"depth={details['depth']:.3f}, "
            f"distance={details['distance']:.3f}, "
            f"overlap={details['overlap']:.3f}"
        )

    output = draw_debug_output(
        image,
        people,
        seats,
        assignments,
    )

    cv2.imwrite(
        OUTPUT_IMAGE_PATH,
        output,
    )

    print(f"\nSaved result: {OUTPUT_IMAGE_PATH}")
    print(f"Saved depth map: {OUTPUT_DEPTH_PATH}")


if __name__ == "__main__":
    main()