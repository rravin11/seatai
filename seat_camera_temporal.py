"""Live-camera scaffold for SeatAI.

Connect the existing segmentation/depth inference functions inside infer_frame().
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2

CAMERA_INDEX = 0
OCCUPIED_CONFIRM_SECONDS = 1.0
AVAILABLE_CONFIRM_SECONDS = 3.0


@dataclass
class SeatMemory:
    stable_state: str = "uncertain"
    candidate_state: str = "uncertain"
    candidate_since: float = field(default_factory=time.monotonic)

    def update(self, observed_state: str, now: float) -> str:
        if observed_state != self.candidate_state:
            self.candidate_state = observed_state
            self.candidate_since = now

        elapsed = now - self.candidate_since

        if observed_state == "occupied":
            required = OCCUPIED_CONFIRM_SECONDS
        elif observed_state == "available":
            required = AVAILABLE_CONFIRM_SECONDS
        else:
            required = 0.5

        if elapsed >= required:
            self.stable_state = observed_state

        return self.stable_state


def infer_frame(frame):
    """Replace with YOLO segmentation + depth + seat matching.

    Return:
        annotated_frame
        observations, for example:
        {
            "seat_left_front": "occupied",
            "seat_back_left": "available",
        }
    """
    return frame, {}


def main() -> None:
    capture = cv2.VideoCapture(CAMERA_INDEX)

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera index {CAMERA_INDEX}. "
            "Try index 1 or 2 for an external camera."
        )

    seat_memories: dict[str, SeatMemory] = {}

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Camera frame could not be read.")
                break

            annotated_frame, observations = infer_frame(frame)
            now = time.monotonic()

            y = 30
            for seat_name, observed_state in observations.items():
                memory = seat_memories.setdefault(seat_name, SeatMemory())
                stable_state = memory.update(observed_state, now)

                cv2.putText(
                    annotated_frame,
                    f"{seat_name}: {stable_state} ({observed_state})",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
                y += 28

            cv2.imshow("SeatAI Live", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
