import cv2
from seat_zones import SEATS, TABLE_BOX

IMAGE_PATH = "/Users/rahulravindranath/Desktop/seatai/.venv/dining1.png"

img = cv2.imread(IMAGE_PATH)

if img is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

# Draw table box
tx1, ty1, tx2, ty2 = TABLE_BOX
cv2.rectangle(img, (tx1, ty1), (tx2, ty2), (255, 0, 0), 3)
cv2.putText(
    img,
    "table",
    (tx1, max(30, ty1 - 10)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (255, 0, 0),
    2
)

# Draw seat zones
for seat_name, box in SEATS.items():
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(
        img,
        seat_name,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

cv2.imshow("Seat Zones", img)
cv2.waitKey(0)
cv2.destroyAllWindows()