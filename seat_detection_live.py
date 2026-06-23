"""
seat_detection_phase1.py
════════════════════════
PHASE 1 prototype — auto-init + dual-window live occupancy

FLOW
  1. INIT    Camera captures one frame. YOLO scans for chairs/seats automatically.
             Detected chair zones are highlighted. Press ENTER to accept or
             adjust with +/- keys, then ENTER to go live.

  2. LIVE    Two windows open simultaneously:
               "Occupancy View"  — clean green (available) / red (occupied) zones
               "YOLO Debug"      — raw YOLO detections with class labels + conf scores

  Occupancy is triggered by:
    • person
    • chair blocked by: laptop, backpack, handbag, suitcase, bottle, cup, book,
                        cell phone, remote, keyboard, mouse, bag

CONTROLS
  INIT
    ENTER      accept detected chairs and go live
    R          re-scan the current frame
    Q          quit

  LIVE
    R          return to init (re-scan)
    S          save snapshot of both windows
    Q          quit

SETUP
  pip install ultralytics opencv-python numpy
"""

import cv2
import math
import time
import collections
import numpy as np
from ultralytics import YOLO

# ─── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "yolo11m.pt"      # or yolov8m.pt
CAMERA_INDEX = 0                 # 0 = phone / default camera
YOLO_CONF    = 0.15
FRAME_SKIP   = 2                 # run YOLO every N frames in live mode

# Chair/seat class names YOLO might detect
CHAIR_CLASSES = {"chair", "couch", "sofa", "bench", "stool"}

# Objects that mark a seat as "claimed / occupied"
PERSON_CLASS  = "person"
CLAIM_CLASSES = {
    "person",
    "laptop", "backpack", "handbag", "suitcase",
    "bottle", "cup", "book",
    "cell phone", "remote", "keyboard", "mouse",
}

# Per-class occupancy weights  (0-1)
CLAIM_WEIGHTS = {
    "person":     1.00,
    "laptop":     0.90,
    "backpack":   0.90,
    "handbag":    0.80,
    "suitcase":   0.85,
    "book":       0.50,
    "bottle":     0.35,
    "cup":        0.30,
    "cell phone": 0.40,
    "remote":     0.30,
    "keyboard":   0.55,
    "mouse":      0.40,
}

# Confidence mins
PERSON_CONF_MIN = 0.45
CLAIM_CONF_MIN  = 0.20
CHAIR_CONF_MIN  = 0.25

# Occupancy decision threshold (smoothed score)
OCCUPIED_THRESHOLD = 0.35

# Temporal smoothing
SMOOTH_WINDOW = 12
EMA_ALPHA     = 0.22

# Zone expansion: grow detected chair box by this fraction on each side
ZONE_EXPAND = 0.12

# Minimum chair box area (px²) to avoid tiny false positives
MIN_CHAIR_AREA = 4000


# ══════════════════════════════════════════════════════════════════════════════
#  Geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def expand_box(box, frac, w, h):
    x1, y1, x2, y2 = box
    pw = (x2 - x1) * frac
    ph = (y2 - y1) * frac
    return (
        int(clamp(x1 - pw, 0, w)),
        int(clamp(y1 - ph, 0, h)),
        int(clamp(x2 + pw, 0, w)),
        int(clamp(y2 + ph, 0, h)),
    )

def box_area(box):
    return max(0, box[2]-box[0]) * max(0, box[3]-box[1])

def intersection_area(a, b):
    ix = max(0, min(a[2],b[2]) - max(a[0],b[0]))
    iy = max(0, min(a[3],b[3]) - max(a[1],b[1]))
    return ix * iy

def iou(a, b):
    inter = intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0

def overlap_ratio_of_zone(zone, obj):
    inter = intersection_area(zone, obj)
    za    = box_area(zone)
    return inter / za if za > 0 else 0.0

def crop_box(img, box):
    x1,y1,x2,y2 = [int(v) for v in box]
    h,w = img.shape[:2]
    return img[clamp(y1,0,h):clamp(y2,0,h), clamp(x1,0,w):clamp(x2,0,w)]


# ══════════════════════════════════════════════════════════════════════════════
#  YOLO helpers
# ══════════════════════════════════════════════════════════════════════════════

def run_yolo(model, frame, conf=YOLO_CONF):
    results = model.predict(source=frame, conf=conf, save=False, verbose=False)
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            label = model.names[int(box.cls[0].item())]
            conf_ = float(box.conf[0].item())
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            dets.append({
                "class": label,
                "confidence": conf_,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
            })
    return dets


def nms_boxes(boxes, iou_thresh=0.45):
    """Simple greedy NMS on a list of (score, box) tuples."""
    boxes = sorted(boxes, key=lambda x: x[0], reverse=True)
    kept  = []
    for score, box in boxes:
        if all(iou(box, k[1]) < iou_thresh for k in kept):
            kept.append((score, box))
    return kept


def detect_chairs(model, frame):
    """
    Run YOLO on frame and return a list of expanded seat zones.
    Falls back to a grid if no chairs are found.
    """
    h, w = frame.shape[:2]
    dets  = run_yolo(model, frame, conf=CHAIR_CONF_MIN)
    chair_boxes = []

    for d in dets:
        if d["class"] not in CHAIR_CLASSES:
            continue
        if box_area(d["bbox"]) < MIN_CHAIR_AREA:
            continue
        chair_boxes.append((d["confidence"], d["bbox"]))

    # NMS to remove duplicate overlapping detections
    chair_boxes = nms_boxes(chair_boxes, iou_thresh=0.40)

    seats = []
    for score, box in chair_boxes:
        zone = expand_box(box, ZONE_EXPAND, w, h)
        cx   = (zone[0]+zone[2])//2
        cy   = (zone[1]+zone[3])//2
        zw   = zone[2]-zone[0]
        zh   = zone[3]-zone[1]
        seats.append({
            "zone":         zone,
            "target":       (cx, cy),
            "sigma_x":      zw * 0.45,
            "sigma_y":      zh * 0.45,
            "anchor_ratio": 0.80,
            "baseline":     frame.copy(),
            "chair_conf":   score,
        })

    # ── Fallback: no chairs detected → divide frame into a 1×N grid ──────────
    if not seats:
        print("  No chairs detected — using full-frame fallback zone.")
        zone = (0, 0, w, h)
        seats.append({
            "zone":         zone,
            "target":       (w//2, h//2),
            "sigma_x":      w * 0.4,
            "sigma_y":      h * 0.4,
            "anchor_ratio": 0.80,
            "baseline":     frame.copy(),
            "chair_conf":   0.0,
        })

    return seats


# ══════════════════════════════════════════════════════════════════════════════
#  Baseline diff
# ══════════════════════════════════════════════════════════════════════════════

def baseline_diff_score(frame, seat, normalizer=55.0):
    cc = crop_box(frame, seat["zone"])
    bc = crop_box(seat["baseline"], seat["zone"])
    if cc.size == 0 or bc.size == 0:
        return 0.0
    if cc.shape != bc.shape:
        bc = cv2.resize(bc, (cc.shape[1], cc.shape[0]))
    diff = cv2.absdiff(
        cv2.cvtColor(cc, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(bc, cv2.COLOR_BGR2GRAY),
    )
    return clamp(diff.mean() / normalizer, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Occupancy scoring
# ══════════════════════════════════════════════════════════════════════════════

def score_detections_for_seat(seat, dets):
    """
    Returns (best_score, best_label, best_bbox) for the highest-scoring
    claim object overlapping this seat zone.
    """
    zone      = seat["zone"]
    best      = 0.0
    best_lbl  = None
    best_bbox = None

    for d in dets:
        label = d["class"]
        if label not in CLAIM_CLASSES:
            continue
        conf = d["confidence"]
        min_conf = PERSON_CONF_MIN if label == PERSON_CLASS else CLAIM_CONF_MIN
        if conf < min_conf:
            continue

        ov     = overlap_ratio_of_zone(zone, d["bbox"])
        weight = CLAIM_WEIGHTS.get(label, 0.25)
        score  = weight * clamp(ov / 0.15, 0, 1) * conf

        if score > best:
            best, best_lbl, best_bbox = score, label, d["bbox"]

    return clamp(best, 0.0, 1.0), best_lbl, best_bbox


def raw_occupancy_score(seat_idx, seats, dets, frame):
    seat       = seats[seat_idx]
    det_score, best_lbl, best_bbox = score_detections_for_seat(seat, dets)
    bd         = baseline_diff_score(frame, seat)

    if det_score >= 0.10:
        occ = clamp(0.60 * det_score + 0.40 * bd, 0, 1)
    else:
        occ = clamp(0.25 * bd, 0, 1)

    return occ, best_lbl, best_bbox


# ══════════════════════════════════════════════════════════════════════════════
#  Temporal smoother
# ══════════════════════════════════════════════════════════════════════════════

class Smoother:
    def __init__(self, n):
        self.buf = [collections.deque(maxlen=SMOOTH_WINDOW) for _ in range(n)]
        self.ema = [None] * n

    def update(self, idx, raw):
        self.buf[idx].append(raw)
        roll = sum(self.buf[idx]) / len(self.buf[idx])
        if self.ema[idx] is None:
            self.ema[idx] = roll
        else:
            self.ema[idx] = EMA_ALPHA * roll + (1 - EMA_ALPHA) * self.ema[idx]
        return self.ema[idx]

    def get(self, idx):
        return self.ema[idx] if self.ema[idx] is not None else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  Drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

GREEN  = (40, 200, 40)
RED    = (40, 40, 200)
CYAN   = (200, 200, 0)
YELLOW = (0, 210, 210)
WHITE  = (230, 230, 230)
PURPLE = (200, 50, 200)


def draw_init_frame(frame, seats):
    out = frame.copy()
    h, w = out.shape[:2]

    for i, seat in enumerate(seats):
        z     = seat["zone"]
        conf  = seat.get("chair_conf", 0.0)
        color = CYAN

        # semi-transparent fill
        ov = out.copy()
        cv2.rectangle(ov, (z[0],z[1]), (z[2],z[3]), color, -1)
        cv2.addWeighted(ov, 0.15, out, 0.85, 0, out)
        cv2.rectangle(out, (z[0],z[1]), (z[2],z[3]), color, 2)

        label = f"Seat {i+1}"
        if conf > 0:
            label += f"  ({conf:.2f})"
        cv2.putText(out, label, (z[0]+6, z[1]+24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    lines = [
        f"INIT  |  {len(seats)} seat(s) auto-detected",
        "ENTER to confirm and go live   |   R to re-scan   |   Q to quit",
    ]
    for i, ln in enumerate(lines):
        cv2.putText(out, ln, (12, 28+i*26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, CYAN, 1)
    return out


def draw_occupancy_view(frame, seats, smoothed_scores, trigger_labels):
    """Clean display window — green/red zones only."""
    out = frame.copy()
    h, w = out.shape[:2]

    for i, seat in enumerate(seats):
        z     = seat["zone"]
        score = smoothed_scores[i]
        occ   = score >= OCCUPIED_THRESHOLD
        color = RED if occ else GREEN
        state = "OCCUPIED" if occ else "AVAILABLE"

        ov = out.copy()
        cv2.rectangle(ov, (z[0],z[1]), (z[2],z[3]), color, -1)
        cv2.addWeighted(ov, 0.22, out, 0.78, 0, out)
        cv2.rectangle(out, (z[0],z[1]), (z[2],z[3]), color, 3)

        lx = (z[0]+z[2])//2
        ly = (z[1]+z[3])//2
        (tw,th),_ = cv2.getTextSize(state, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.putText(out, state, (lx-tw//2, ly+th//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        cv2.putText(out, f"Seat {i+1}", (z[0]+6, z[1]+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)

        # what triggered occupancy
        trig = trigger_labels.get(i)
        if trig and occ:
            cv2.putText(out, f"[{trig}]", (z[0]+6, z[3]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, YELLOW, 1)

    cv2.putText(out, "[R] re-init   [S] snapshot   [Q] quit",
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1)
    return out


def draw_debug_view(frame, seats, dets, smoothed_scores):
    """YOLO debug window — raw detections + seat scores."""
    out = frame.copy()
    h, w = out.shape[:2]

    # All YOLO detections
    for d in dets:
        x1,y1,x2,y2 = d["bbox"]
        label = d["class"]
        conf  = d["confidence"]

        if label == PERSON_CLASS:
            color = (255, 100, 0)
        elif label in CLAIM_CLASSES:
            color = PURPLE
        else:
            color = (120, 120, 120)

        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        cv2.putText(out, f"{label} {conf:.2f}",
                    (x1, max(14, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1)

    # Seat zones with raw score
    for i, seat in enumerate(seats):
        z     = seat["zone"]
        score = smoothed_scores[i]
        occ   = score >= OCCUPIED_THRESHOLD
        color = RED if occ else GREEN
        cv2.rectangle(out, (z[0],z[1]), (z[2],z[3]), color, 2)
        cv2.putText(out, f"S{i+1} score={score:.2f}",
                    (z[0]+4, z[3]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)

    cv2.putText(out, "YOLO DEBUG VIEW",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    cv2.putText(out, f"detections: {len(dets)}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera {CAMERA_INDEX}.\n"
            "Try changing CAMERA_INDEX (0, 1, 2 …)\n"
            "and check System Settings -> Privacy -> Camera."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    print("Loading YOLO model …")
    model = YOLO(MODEL_PATH)

    # ── state ────────────────────────────────────────────────────────────────
    phase         = "init"
    seats         = []
    smoother      = None
    last_dets     = []
    last_scores   = []
    last_triggers = {}
    frame_count   = 0
    fps           = 0.0
    t_prev        = time.time()
    init_frame    = None
    flash_msg     = ""
    flash_until   = 0.0

    print("Point camera at the scene and press ENTER to auto-detect chairs.")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        frame_count += 1
        t_now  = time.time()
        fps    = 0.9*fps + 0.1/max(t_now-t_prev, 1e-6)
        t_prev = t_now

        # ── INIT phase ───────────────────────────────────────────────────────
        if phase == "init":
            vis = frame.copy()

            if seats:
                vis = draw_init_frame(frame, seats)
            else:
                # Guide text before first scan
                cv2.putText(vis,
                    "Point camera at the full scene, then press ENTER to auto-detect chairs",
                    (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.58, CYAN, 2)
                cv2.putText(vis, "Q to quit",
                    (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.50, WHITE, 1)

            if t_now < flash_until:
                cv2.putText(vis, flash_msg,
                    (vis.shape[1]//2-200, vis.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, YELLOW, 2)

            cv2.imshow("Seat Detection — Init", vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == 13:  # ENTER
                if not seats:
                    # First ENTER: scan the frame
                    print("  Scanning for chairs …")
                    init_frame = frame.copy()
                    seats      = detect_chairs(model, init_frame)
                    print(f"  Found {len(seats)} seat(s).")
                    if not seats:
                        flash_msg   = "No chairs found — try re-scanning (R)"
                        flash_until = t_now + 2.5
                else:
                    # Second ENTER: confirm and go live
                    smoother      = Smoother(len(seats))
                    last_scores   = [0.0] * len(seats)
                    last_triggers = {}
                    phase         = "live"
                    cv2.destroyWindow("Seat Detection — Init")
                    print(f"\nLIVE — tracking {len(seats)} seat(s). Two windows open.")

            elif key == ord('r'):
                # Re-scan
                print("  Re-scanning …")
                init_frame = frame.copy()
                seats      = detect_chairs(model, init_frame)
                print(f"  Found {len(seats)} seat(s).")
                flash_msg   = f"Re-scanned: {len(seats)} seat(s) found"
                flash_until = t_now + 1.5

        # ── LIVE phase ───────────────────────────────────────────────────────
        else:
            if frame_count % FRAME_SKIP == 0:
                last_dets = run_yolo(model, frame)

                new_scores   = []
                new_triggers = {}
                for i, seat in enumerate(seats):
                    raw, trig_lbl, _ = raw_occupancy_score(i, seats, last_dets, frame)
                    s = smoother.update(i, raw)
                    new_scores.append(s)
                    if trig_lbl:
                        new_triggers[i] = trig_lbl
                last_scores   = new_scores
                last_triggers = new_triggers

            occ_vis   = draw_occupancy_view(frame, seats, last_scores, last_triggers)
            debug_vis = draw_debug_view(frame, seats, last_dets, last_scores)

            if t_now < flash_until:
                cv2.putText(occ_vis, flash_msg,
                    (occ_vis.shape[1]//2-160, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, YELLOW, 2)

            cv2.imshow("Occupancy View", occ_vis)
            cv2.imshow("YOLO Debug",     debug_vis)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('r'):
                # back to init
                seats         = []
                smoother      = None
                last_dets     = []
                last_scores   = []
                last_triggers = {}
                phase         = "init"
                cv2.destroyWindow("Occupancy View")
                cv2.destroyWindow("YOLO Debug")
                print("\nBack to init.")

            elif key == ord('s'):
                ts = int(time.time())
                cv2.imwrite(f"occupancy_{ts}.png", occ_vis)
                cv2.imwrite(f"debug_{ts}.png",     debug_vis)
                print(f"Snapshots saved: occupancy_{ts}.png  debug_{ts}.png")
                flash_msg   = "Snapshots saved!"
                flash_until = t_now + 1.5

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()