import cv2

SEATS = {
    "seat_left_front": {
        "zone": [30, 260, 430, 835],
        "target": (320, 610),
        "sigma_x": 120,
        "sigma_y": 95,
        "anchor_ratio": 0.68,
    },
    "seat_back_left": {
        "zone": [430, 120, 760, 430],
        "target": (620, 250),
        "sigma_x": 110,
        "sigma_y": 70,
        "anchor_ratio": 0.70,
    },
    "seat_back_right": {
        "zone": [860, 140, 1150, 560],
        "target": (1035, 410),
        "sigma_x": 105,
        "sigma_y": 80,
        "anchor_ratio": 0.45,
    },
    "seat_right_front": {
        "zone": [860, 430, 1140, 900],
        "target": (980, 720),
        "sigma_x": 115,
        "sigma_y": 95,
        "anchor_ratio": 0.72,
    },
}

TABLE_BOX = [270, 182, 920, 760]

IMAGE_PATH = "/Users/rahulravindranath/Desktop/seatai/.venv/dining1.png"
OUTPUT_PATH = "seat_zones_output.png"


def draw_seat_zones(image_path=IMAGE_PATH, output_path=OUTPUT_PATH):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

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

    for seat_name, seat_info in SEATS.items():
        x1, y1, x2, y2 = seat_info["zone"]
        tx, ty = seat_info["target"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            img,
            seat_name,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.circle(img, (tx, ty), 7, (0, 0, 255), -1)

    cv2.imwrite(output_path, img)
    print(f"Saved seat zone image to: {output_path}")



if __name__ == "__main__":
    draw_seat_zones()