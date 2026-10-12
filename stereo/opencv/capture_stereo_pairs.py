"""Preview two cameras and save matched ChArUco calibration image pairs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


TOOL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TOOL_ROOT.parents[1]
WINDOW_NAME = "Stereo calibration capture"


def backend_id(name: str) -> int:
    return {
        "auto": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "v4l2": cv2.CAP_V4L2,
    }[name]


def open_camera(
    index: int, backend: str, width: int, height: int, fps: float, fourcc: str
) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(index, backend_id(backend))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera {index} with backend {backend}")
    if fourcc:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def grab_pair(
    left: cv2.VideoCapture, right: cv2.VideoCapture
) -> tuple[np.ndarray, np.ndarray]:
    # Grab both before decoding to reduce software synchronization skew.
    if not left.grab() or not right.grab():
        raise RuntimeError("Failed to grab one or both cameras")
    left_ok, left_frame = left.retrieve()
    right_ok, right_frame = right.retrieve()
    if not left_ok or not right_ok:
        raise RuntimeError("Failed to retrieve one or both camera frames")
    return left_frame, right_frame


def next_index(left_dir: Path, right_dir: Path) -> int:
    def indexes(directory: Path) -> set[int]:
        return {
            int(item.stem)
            for item in directory.glob("*.png")
            if item.stem.isdecimal()
        }

    left = indexes(left_dir)
    right = indexes(right_dir)
    if left != right:
        raise RuntimeError("Existing left/right images are not paired")
    return max(left | right, default=0) + 1


def fit(image: np.ndarray, max_width: int = 1800, max_height: int = 950) -> np.ndarray:
    scale = min(max_width / image.shape[1], max_height / image.shape[0], 1.0)
    return image if scale >= 1 else cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )


def main() -> int:
    default_backend = "dshow" if os.name == "nt" else "v4l2"
    parser = argparse.ArgumentParser(description="Capture paired stereo images")
    parser.add_argument("--left", type=int, default=0)
    parser.add_argument("--right", type=int, default=1)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument(
        "--backend", choices=["auto", "dshow", "msmf", "v4l2"], default=default_backend
    )
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--output-dir", default="data/calibration")
    args = parser.parse_args()
    if args.left == args.right:
        raise ValueError("Left and right camera indexes must differ")
    if min(args.width, args.height, args.fps) <= 0:
        raise ValueError("Width, height and FPS must be positive")
    if len(args.fourcc) not in (0, 4):
        raise ValueError("fourcc must be empty or exactly four characters")

    output_root = Path(args.output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    left_dir, right_dir = output_root / "left", output_root / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    pair_index = next_index(left_dir, right_dir)

    left = open_camera(
        args.left, args.backend, args.width, args.height, args.fps, args.fourcc
    )
    try:
        right = open_camera(
            args.right, args.backend, args.width, args.height, args.fps, args.fourcc
        )
    except Exception:
        left.release()
        raise
    try:
        for _ in range(args.warmup_frames):
            grab_pair(left, right)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        while True:
            left_frame, right_frame = grab_pair(left, right)
            if left_frame.shape != right_frame.shape:
                raise RuntimeError(
                    f"Camera frame shapes differ: {left_frame.shape} vs {right_frame.shape}"
                )
            if left_frame.shape[:2] != (args.height, args.width):
                raise RuntimeError(
                    f"Requested {args.width}x{args.height}, but cameras returned "
                    f"{left_frame.shape[1]}x{left_frame.shape[0]}; "
                    "use a supported resolution matching your intrinsic CSV files"
                )
            left_view, right_view = left_frame.copy(), right_frame.copy()
            cv2.putText(left_view, "LEFT", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(right_view, "RIGHT", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            canvas = np.hstack((left_view, right_view))
            cv2.putText(
                canvas,
                f"SPACE/S save pair {pair_index:06d}   Q/ESC quit",
                (18, canvas.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow(WINDOW_NAME, fit(canvas))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord(" "), ord("s"), ord("S")):
                name = f"{pair_index:06d}.png"
                left_path, right_path = left_dir / name, right_dir / name
                if left_path.exists() or right_path.exists():
                    raise FileExistsError(f"Refusing to overwrite pair {name}")
                if not cv2.imwrite(str(left_path), left_frame):
                    raise OSError(f"Cannot write {left_path}")
                if not cv2.imwrite(str(right_path), right_frame):
                    left_path.unlink(missing_ok=True)
                    raise OSError(f"Cannot write {right_path}")
                print(f"Saved pair {pair_index:06d}")
                pair_index += 1
    finally:
        left.release()
        right.release()
        cv2.destroyAllWindows()
    print(f"Images: {output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
