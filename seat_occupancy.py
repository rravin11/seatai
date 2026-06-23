import os
import cv2
from ultralytics import YOLO
from seat_zones import SEATS, TABLE_BOX

IMAGE_PATHS = [
    "/Users/rahulravindranath/Desktop/seatai/dining.png",
    "/Users/rahulravindranath/Desktop/seatai/dining1.png",
    "/Users/rahulravindranath/Desktop/seatai/people_dining1.png",
]

MODEL_PATH = "yolo11m.pt"
OUTPUT_DIR = "/Users/rahulravindranath/Desktop/seatai/outputs"

PERSON_CONF_MIN = 0.50
OTHER_CONF_MIN = 0.15


def point_in_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def squared_distance(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


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

            if label == "person" and conf < PERSON_CONF_MIN:
                continue
            if label != "person" and conf < OTHER_CONF_MIN:
                continue

            detections.append({
                "class": label,
                "confidence": conf,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })

    return detections


def clear_debug_assignments():
    for seat_info in SEATS.values():
        if "_assigned_anchor" in seat_info:
            del seat_info["_assigned_anchor"]


def assign_seat_states(seats, detections, max_distance=220):
    seat_states = {seat_name: "available" for seat_name in seats}
    taken_seats = set()

    for det in detections:
        if det["class"] != "person":
            continue

        x1, y1, x2, y2 = det["bbox"]

        best_seat = None
        best_dist = float("inf")
        best_anchor = None

        for seat_name, seat_info in seats.items():
            if seat_name in taken_seats:
                continue

            seat_box = seat_info["zone"]
            anchor_ratio = seat_info.get("anchor_ratio", 0.70)

            seat_anchor = (
                (x1 + x2) // 2,
                int(y1 + anchor_ratio * (y2 - y1))
            )

            if not point_in_box(seat_anchor, seat_box):
                continue

            center = box_center(seat_box)
            dist = squared_distance(seat_anchor, center)

            if dist < best_dist:
                best_dist = dist
                best_seat = seat_name
                best_anchor = seat_anchor

        if best_seat is not None and best_dist <= max_distance ** 2:
            seat_states[best_seat] = "occupied"
            taken_seats.add(best_seat)
            seats[best_seat]["_assigned_anchor"] = best_anchor

    return seat_states


def draw_results(img, seats, seat_states, detections):
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
            0.6,
            (128, 0, 128),
            2
        )

    for seat_name, seat_info in seats.items():
        seat_box = seat_info["zone"]
        target = seat_info.get("target", None)

        x1, y1, x2, y2 = seat_box
        state = seat_states[seat_name]

        color = (0, 255, 0) if state == "available" else (0, 0, 255)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            img,
            f"{seat_name}: {state}",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

        cx, cy = box_center(seat_box)
        cv2.circle(img, (cx, cy), 5, (255, 255, 0), -1)

        if target is not None:
            cv2.circle(img, target, 7, (0, 165, 255), -1)

        if "_assigned_anchor" in seat_info:
            ax, ay = seat_info["_assigned_anchor"]
            cv2.circle(img, (ax, ay), 6, (0, 255, 255), -1)
            if target is not None:
                cv2.line(img, (ax, ay), target, (255, 255, 255), 2)

    return img


def process_image(model, image_path):
    clear_debug_assignments()

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return

    results = model.predict(
        source=image_path,
        conf=0.10,
        save=False
    )

    detections = extract_detections(results, model)

    print(f"\n=== Processing: {image_path} ===")
    print("\nDetections:")
    for det in detections:
        print(det)

    seat_states = assign_seat_states(SEATS, detections, max_distance=220)

    print("\nSeat states:")
    for seat_name, state in seat_states.items():
        print(f"{seat_name}: {state}")

    output = draw_results(img, SEATS, seat_states, detections)

    base_name = os.path.basename(image_path)
    file_stem, _ = os.path.splitext(base_name)
    output_path = os.path.join(OUTPUT_DIR, f"{file_stem}_seat_occupancy_output.png")

    cv2.imwrite(output_path, output)
    print(f"\nSaved result to: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = YOLO(MODEL_PATH)

    for image_path in IMAGE_PATHS:
        process_image(model, image_path)


if __name__ == "__main__":
    main()