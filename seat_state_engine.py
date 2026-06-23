import cv2
import math
from ultralytics import YOLO
from seat_zones import SEATS, TABLE_BOX

IMAGE_PATH = "/Users/rahulravindranath/Desktop/seatai/.venv/people_dining1.png"
BASELINE_EMPTY_IMAGE_PATH = "/Users/rahulravindranath/Desktop/seatai/.venv/dining1.png"
MODEL_PATH = "yolo11m.pt"

YOLO_CONF = 0.10

PERSON_CLASS = "person"
CLAIM_OBJECT_CLASSES = {"backpack", "handbag", "laptop", "bottle", "cup"}

OCCUPIED_THRESHOLD = 0.42
CLAIMED_THRESHOLD = 0.35
AVAILABLE_THRESHOLD = 0.55
PERSON_MATCH_THRESHOLD = 0.28
BASELINE_DIFF_NORMALIZER = 80.0

# class-specific cleanup thresholds
PERSON_CONF_MIN = 0.50
OTHER_CONF_MIN = 0.15


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def point_in_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def intersection_area(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    return inter_w * inter_h


def overlap_ratio_of_zone(zone_box, obj_box):
    inter = intersection_area(zone_box, obj_box)
    zone_a = box_area(zone_box)
    return inter / zone_a if zone_a > 0 else 0.0


def crop_box(img, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = img.shape[:2]

    x1 = clamp(x1, 0, w - 1)
    x2 = clamp(x2, 0, w - 1)
    y1 = clamp(y1, 0, h - 1)
    y2 = clamp(y2, 0, h - 1)

    return img[y1:y2, x1:x2]


def seat_specific_anchor(person_box, anchor_ratio):
    x1, y1, x2, y2 = person_box
    return ((x1 + x2) / 2.0, y1 + anchor_ratio * (y2 - y1))


def gaussian_score(point, target, sigma_x, sigma_y):
    px, py = point
    tx, ty = target

    if sigma_x <= 0 or sigma_y <= 0:
        return 0.0

    dx = px - tx
    dy = py - ty

    exponent = -(
        (dx * dx) / (2.0 * sigma_x * sigma_x) +
        (dy * dy) / (2.0 * sigma_y * sigma_y)
    )
    return math.exp(exponent)


def vertical_depth_score(anchor_y, target_y, sigma_y):
    if sigma_y <= 0:
        return 0.0

    dy = abs(anchor_y - target_y)
    score = math.exp(-(dy * dy) / (2.0 * sigma_y * sigma_y))
    return clamp(score, 0.0, 1.0)


def extract_detections(results, model):
    detections = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = model.names[cls_id]

            # class-specific filtering
            if label == PERSON_CLASS and conf < PERSON_CONF_MIN:
                continue
            if label != PERSON_CLASS and conf < OTHER_CONF_MIN:
                continue

            detections.append({
                "class": label,
                "confidence": conf,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })

    return detections


def seat_baseline_difference_score(current_img, baseline_img, zone_box):
    cur_crop = crop_box(current_img, zone_box)
    base_crop = crop_box(baseline_img, zone_box)

    if cur_crop.size == 0 or base_crop.size == 0:
        return 0.0

    if cur_crop.shape != base_crop.shape:
        base_crop = cv2.resize(base_crop, (cur_crop.shape[1], cur_crop.shape[0]))

    cur_gray = cv2.cvtColor(cur_crop, cv2.COLOR_BGR2GRAY)
    base_gray = cv2.cvtColor(base_crop, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(cur_gray, base_gray)
    mean_diff = float(diff.mean())

    score = mean_diff / BASELINE_DIFF_NORMALIZER
    return clamp(score, 0.0, 1.0)


def compute_claim_object_score_for_seat(zone_box, detections):
    best_score = 0.0

    for det in detections:
        label = det["class"]
        if label not in CLAIM_OBJECT_CLASSES:
            continue

        overlap = overlap_ratio_of_zone(zone_box, det["bbox"])

        class_weight = {
            "backpack": 1.00,
            "handbag": 0.85,
            "laptop": 0.75,
            "bottle": 0.35,
            "cup": 0.30,
        }.get(label, 0.25)

        score = class_weight * clamp(overlap / 0.20, 0.0, 1.0) * det["confidence"]
        best_score = max(best_score, score)

    return clamp(best_score, 0.0, 1.0)


def compute_person_seat_match_score(person_det, seat_info):
    zone = seat_info["zone"]
    target = seat_info["target"]
    sigma_x = seat_info["sigma_x"]
    sigma_y = seat_info["sigma_y"]
    anchor_ratio = seat_info["anchor_ratio"]

    pbox = person_det["bbox"]
    anchor = seat_specific_anchor(pbox, anchor_ratio)

    inside = 1.0 if point_in_box(anchor, zone) else 0.0
    overlap = overlap_ratio_of_zone(zone, pbox)
    g_score = gaussian_score(anchor, target, sigma_x, sigma_y)
    v_score = vertical_depth_score(anchor[1], target[1], sigma_y)

    score = (
        0.50 * g_score +
        0.20 * v_score +
        0.15 * clamp(overlap / 0.25, 0.0, 1.0) +
        0.15 * inside
    )

    return clamp(score, 0.0, 1.0), anchor


def greedy_assign_people_to_seats(seats, detections):
    people = [d for d in detections if d["class"] == PERSON_CLASS]
    candidates = []

    for p_idx, det in enumerate(people):
        for seat_name, seat_info in seats.items():
            score, anchor = compute_person_seat_match_score(det, seat_info)
            if score >= PERSON_MATCH_THRESHOLD:
                candidates.append((score, p_idx, seat_name, anchor))

    candidates.sort(reverse=True, key=lambda x: x[0])

    assignments = {}
    used_people = set()
    used_seats = set()

    for score, p_idx, seat_name, anchor in candidates:
        if p_idx in used_people:
            continue
        if seat_name in used_seats:
            continue

        assignments[seat_name] = {
            "person_det": people[p_idx],
            "match_score": score,
            "anchor": anchor
        }
        used_people.add(p_idx)
        used_seats.add(seat_name)

    return assignments


def score_single_seat(seat_name, seat_info, assignments, detections, current_img, baseline_img):
    zone = seat_info["zone"]

    baseline_diff = seat_baseline_difference_score(current_img, baseline_img, zone)
    claim_raw = compute_claim_object_score_for_seat(zone, detections)

    person_assigned = seat_name in assignments
    person_match = assignments[seat_name]["match_score"] if person_assigned else 0.0

    occupied_score = 0.0
    if person_assigned:
        occupied_score = 0.70 * person_match + 0.30 * baseline_diff
    occupied_score = clamp(occupied_score, 0.0, 1.0)

    claimed_score = 0.0
    if not person_assigned:
        claimed_score = 0.70 * claim_raw + 0.30 * baseline_diff
    claimed_score = clamp(claimed_score, 0.0, 1.0)

    no_person = 0.0 if person_assigned else 1.0
    no_claim = 1.0 - claim_raw
    emptiness = 1.0 - baseline_diff

    available_score = 0.40 * no_person + 0.30 * no_claim + 0.30 * emptiness
    available_score = clamp(available_score, 0.0, 1.0)

    strongest = max(occupied_score, claimed_score, available_score)
    uncertain_score = clamp(1.0 - strongest, 0.0, 1.0)

    if occupied_score >= OCCUPIED_THRESHOLD and occupied_score >= claimed_score and occupied_score >= available_score:
        state = "occupied"
    elif claimed_score >= CLAIMED_THRESHOLD and claimed_score >= available_score:
        state = "claimed"
    elif available_score >= AVAILABLE_THRESHOLD:
        state = "available"
    else:
        state = "uncertain"

    return {
        "state": state,
        "occupied_score": occupied_score,
        "claimed_score": claimed_score,
        "available_score": available_score,
        "uncertain_score": uncertain_score,
        "baseline_diff": baseline_diff,
        "person_assigned": person_assigned,
        "person_match": person_match,
    }


def evaluate_all_seats(seats, detections, current_img, baseline_img):
    assignments = greedy_assign_people_to_seats(seats, detections)
    seat_results = {}

    for seat_name, seat_info in seats.items():
        seat_results[seat_name] = score_single_seat(
            seat_name,
            seat_info,
            assignments,
            detections,
            current_img,
            baseline_img
        )

    return seat_results, assignments


def draw_results(img, seats, seat_results, detections, assignments):
    tx1, ty1, tx2, ty2 = TABLE_BOX
    cv2.rectangle(img, (tx1, ty1), (tx2, ty2), (255, 0, 0), 2)
    cv2.putText(
        img,
        "table",
        (tx1, max(30, ty1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["class"]
        conf = det["confidence"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (128, 0, 128), 2)
        cv2.putText(
            img,
            f"{label} {conf:.2f}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (128, 0, 128),
            2
        )

    for seat_name, seat_info in seats.items():
        zone = seat_info["zone"]
        target = seat_info["target"]
        result = seat_results[seat_name]

        x1, y1, x2, y2 = zone

        if result["state"] == "occupied":
            color = (0, 0, 255)
        elif result["state"] == "claimed":
            color = (0, 165, 255)
        elif result["state"] == "available":
            color = (0, 255, 0)
        else:
            color = (180, 180, 180)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.circle(img, target, 7, (255, 255, 0), -1)

        cv2.putText(
            img,
            f"{seat_name}: {result['state']}",
            (x1, max(28, y1 - 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )
        cv2.putText(
            img,
            f"O:{result['occupied_score']:.2f} C:{result['claimed_score']:.2f} A:{result['available_score']:.2f}",
            (x1, max(48, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2
        )

    for seat_name, assignment in assignments.items():
        target = seats[seat_name]["target"]
        ax, ay = assignment["anchor"]

        cv2.circle(img, (int(ax), int(ay)), 6, (0, 255, 255), -1)
        cv2.line(img, (int(ax), int(ay)), target, (255, 255, 255), 2)

    return img

def debug_person_seat_scores(seats, detections):
    people = [d for d in detections if d["class"] == PERSON_CLASS]

    print("\nPerson-to-seat match scores:")
    for i, det in enumerate(people):
        print(f"\nPerson {i}: bbox={det['bbox']}, conf={det['confidence']:.2f}")
        for seat_name, seat_info in seats.items():
            score, anchor = compute_person_seat_match_score(det, seat_info)
            print(
                f"  {seat_name}: score={score:.3f}, "
                f"anchor=({anchor[0]:.1f}, {anchor[1]:.1f})"
            )
            
def main():
    current_img = cv2.imread(IMAGE_PATH)
    if current_img is None:
        raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

    baseline_img = cv2.imread(BASELINE_EMPTY_IMAGE_PATH)
    if baseline_img is None:
        raise FileNotFoundError(f"Could not load baseline image: {BASELINE_EMPTY_IMAGE_PATH}")

    model = YOLO(MODEL_PATH)
    results = model.predict(source=IMAGE_PATH, conf=YOLO_CONF, save=False)
    detections = extract_detections(results, model)

    print("\nDetections:")
    for det in detections:
        print(det)

    seat_results, assignments = evaluate_all_seats(SEATS, detections, current_img, baseline_img)
    debug_person_seat_scores(SEATS, detections)

    print("\nSeat results:")
    for seat_name, result in seat_results.items():
        print(
            f"{seat_name}: {result['state']} | "
            f"O={result['occupied_score']:.2f}, "
            f"C={result['claimed_score']:.2f}, "
            f"A={result['available_score']:.2f}"
        )

    output = current_img.copy()
    output = draw_results(output, SEATS, seat_results, detections, assignments)

    output_path = "seat_state_engine_output.png"
    cv2.imwrite(output_path, output)
    print(f"\nSaved result to: {output_path}")


if __name__ == "__main__":
    main()