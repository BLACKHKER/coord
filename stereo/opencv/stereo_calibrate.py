#!/usr/bin/env python3
"""Calibrate a ChArUco stereo pair and write the YAML used by stereo_fusion."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CALIBRATION_WINDOW = "ChArUco stereo calibration review"
TOOL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TOOL_ROOT.parents[1]
DEFAULT_CONFIG = TOOL_ROOT / "stereo_calibrate_charuco.yml"


def config_value(storage: cv2.FileStorage, key: str, default):
    node = storage.getNode(key)
    if node.empty():
        return default
    if isinstance(default, str):
        return node.string()
    if isinstance(default, int):
        return int(node.real())
    return float(node.real())


def read_config(path: Path) -> dict:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise ValueError(f"Cannot open calibration config: {path}")
    try:
        return {
            "left_dir": Path(config_value(storage, "left_dir", "data/calibration/left")),
            "right_dir": Path(config_value(storage, "right_dir", "data/calibration/right")),
            "output": Path(config_value(storage, "output", "stereo.yml")),
            "left_intrinsics_csv": Path(config_value(storage, "left_intrinsics_csv", "")),
            "right_intrinsics_csv": Path(config_value(storage, "right_intrinsics_csv", "")),
            "board_type": config_value(storage, "board_type", "charuco"),
            "cols": config_value(storage, "cols", 10),
            "rows": config_value(storage, "rows", 7),
            "square_size_m": config_value(storage, "square_size_m", 0.068),
            "marker_size_m": config_value(storage, "marker_size_m", 0.0535),
            "aruco_dictionary": config_value(storage, "aruco_dictionary", "DICT_6X6_50"),
            "preview_mode": config_value(storage, "preview_mode", "off"),
            "preview_delay_ms": config_value(storage, "preview_delay_ms", 500),
        }
    finally:
        storage.release()


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"Image directory does not exist: {directory}")
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def paired_image_files(left_dir: Path, right_dir: Path) -> tuple[list[Path], list[Path]]:
    left = {path.name: path for path in image_files(left_dir)}
    right = {path.name: path for path in image_files(right_dir)}
    if not left or left.keys() != right.keys():
        raise ValueError(
            "Left/right images must have identical non-empty filename sets. "
            f"Only left={sorted(left.keys() - right.keys())}; "
            f"only right={sorted(right.keys() - left.keys())}"
        )
    names = sorted(left)
    return [left[name] for name in names], [right[name] for name in names]


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_intrinsics_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if values.shape != (5, 3) or not np.all(np.isfinite(values)):
        raise ValueError(f"Intrinsic CSV must be a finite 5x3 matrix: {path}")
    camera_matrix = values[:3, :3].copy()
    # Match world/camera.py: MATLAB exports the transposed intrinsic matrix.
    if "MATLAB" in path.name:
        camera_matrix = camera_matrix.T.copy()
    if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0:
        raise ValueError(f"Invalid camera matrix in {path}")
    if not np.allclose(camera_matrix[2], [0, 0, 1]):
        raise ValueError(f"Expected OpenCV intrinsic matrix (or a MATLAB-named CSV): {path}")
    distortion = np.asarray(
        [[values[3, 0], values[3, 1], values[4, 0], values[4, 1], values[3, 2]]],
        dtype=np.float64,
    )
    return camera_matrix, distortion


def dictionary_from_name(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco is unavailable; install opencv-contrib-python")
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def create_charuco_board(cols: int, rows: int, square_size: float, marker_size: float, dictionary):
    """Create a board across OpenCV 4.5 and newer Python bindings."""
    size = (cols + 1, rows + 1)
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        return cv2.aruco.CharucoBoard_create(
            size[0], size[1], square_size, marker_size, dictionary
        )
    return cv2.aruco.CharucoBoard(size, square_size, marker_size, dictionary)


def detect_charuco(detector, board, image, dictionary):
    """Return (corners, ids), using CharucoDetector or the legacy API."""
    if detector is not None:
        corners, ids, _, _ = detector.detectBoard(image)
        return corners, ids
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(image, dictionary)
    if marker_ids is None:
        return None, None
    count, corners, ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, image, board
    )
    if count <= 0 or corners is None or ids is None:
        return None, None
    return corners, ids


def calibration_preview(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    corners_left,
    corners_right,
    ids_left,
    ids_right,
    common_count: int,
    pair_acceptable: bool,
    left_name: str,
    right_name: str,
    pair_number: int,
    pair_count: int,
    mode: str,
    delay_ms: int,
) -> str:
    left_view = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
    right_view = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2BGR)
    if corners_left is not None and ids_left is not None:
        cv2.aruco.drawDetectedCornersCharuco(left_view, corners_left, ids_left)
    if corners_right is not None and ids_right is not None:
        cv2.aruco.drawDetectedCornersCharuco(right_view, corners_right, ids_right)

    font = cv2.FONT_HERSHEY_SIMPLEX
    content_height = 830
    canvas_width = 1800
    status_height = 120

    def crop_board(image, corners):
        if corners is None or len(corners) == 0:
            return image
        points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        minimum = np.floor(points.min(axis=0)).astype(int)
        maximum = np.ceil(points.max(axis=0)).astype(int)
        board_width = max(1, maximum[0] - minimum[0])
        board_height = max(1, maximum[1] - minimum[1])
        margin_x = max(50, round(board_width * 0.18))
        margin_y = max(50, round(board_height * 0.18))
        x0 = max(0, minimum[0] - margin_x)
        y0 = max(0, minimum[1] - margin_y)
        x1 = min(image.shape[1], maximum[0] + margin_x + 1)
        y1 = min(image.shape[0], maximum[1] + margin_y + 1)
        return image[y0:y1, x0:x1]

    def make_panel(image, corners, side, name, ids, width, zoomed):
        source = crop_board(image, corners) if zoomed else image
        panel = np.zeros((content_height, width, 3), dtype=np.uint8)
        scale = min(width / source.shape[1], content_height / source.shape[0])
        resized = cv2.resize(
            source,
            (max(1, round(source.shape[1] * scale)), max(1, round(source.shape[0] * scale))),
            interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
        )
        x = (width - resized.shape[1]) // 2
        y = (content_height - resized.shape[0]) // 2
        panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        count = 0 if ids is None else len(ids)
        color = (0, 220, 0) if count >= 4 else (0, 0, 255)
        cv2.rectangle(panel, (0, 0), (width, 84), (0, 0, 0), -1)
        cv2.putText(panel, f"{side}: {count} corners", (18, 34), font, 0.82, color, 2)
        cv2.putText(panel, name, (18, 68), font, 0.58, (255, 255, 255), 2)
        return panel

    view_mode = "split"
    zoomed = True

    def make_canvas():
        if view_mode == "left":
            content = make_panel(
                left_view, corners_left, "LEFT", left_name, ids_left, canvas_width, zoomed
            )
        elif view_mode == "right":
            content = make_panel(
                right_view, corners_right, "RIGHT", right_name, ids_right, canvas_width, zoomed
            )
        else:
            panel_width = canvas_width // 2
            content = np.hstack((
                make_panel(
                    left_view, corners_left, "LEFT", left_name, ids_left, panel_width, zoomed
                ),
                make_panel(
                    right_view, corners_right, "RIGHT", right_name, ids_right, panel_width, zoomed
                ),
            ))
        canvas = np.zeros((content_height + status_height, canvas_width, 3), dtype=np.uint8)
        canvas[:content_height] = content
        status = "ACCEPTABLE" if pair_acceptable else "AUTO REJECTED"
        zoom_text = "BOARD ZOOM" if zoomed else "FULL IMAGE"
        cv2.putText(
            canvas,
            f"Pair {pair_number}/{pair_count}: {status}, common={common_count}, {zoom_text}",
            (18, content_height + 40), font, 0.72, (255, 255, 255), 2,
        )
        cv2.putText(
            canvas,
            "1: left  2: right  3: split  Z: zoom  N/SPACE: keep  D: delete  B: back  F: fullscreen  Q: abort",
            (18, content_height + 88), font, 0.62, (220, 220, 220), 2,
        )
        return canvas

    while True:
        cv2.imshow(CALIBRATION_WINDOW, make_canvas())
        key = cv2.waitKey(0 if mode == "key" else delay_ms) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            return "abort"
        if key in (ord("d"), ord("D")):
            return "delete"
        if key in (ord("b"), ord("B")):
            return "back"
        if key == ord("1"):
            view_mode = "left"
            continue
        if key == ord("2"):
            view_mode = "right"
            continue
        if key == ord("3"):
            view_mode = "split"
            continue
        if key in (ord("z"), ord("Z")):
            zoomed = not zoomed
            continue
        if key in (ord("f"), ord("F")):
            is_fullscreen = cv2.getWindowProperty(
                CALIBRATION_WINDOW, cv2.WND_PROP_FULLSCREEN
            ) == cv2.WINDOW_FULLSCREEN
            cv2.setWindowProperty(
                CALIBRATION_WINDOW,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_NORMAL if is_fullscreen else cv2.WINDOW_FULLSCREEN,
            )
            if is_fullscreen:
                cv2.resizeWindow(CALIBRATION_WINDOW, 1800, 950)
            continue
        if mode == "time" and key == 255:
            return "keep"
        if key in (ord("n"), ord("N"), ord(" "), 13):
            return "keep"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--left-dir")
    parser.add_argument("--right-dir")
    parser.add_argument("--left-intrinsics-csv")
    parser.add_argument("--right-intrinsics-csv")
    parser.add_argument("--output")
    parser.add_argument("--preview-mode", choices=("off", "key", "time"))
    parser.add_argument("--preview-delay-ms", type=int)
    args = parser.parse_args()

    settings = read_config(project_path(Path(args.config)))
    if args.left_dir:
        settings["left_dir"] = Path(args.left_dir)
    if args.right_dir:
        settings["right_dir"] = Path(args.right_dir)
    if args.left_intrinsics_csv:
        settings["left_intrinsics_csv"] = Path(args.left_intrinsics_csv)
    if args.right_intrinsics_csv:
        settings["right_intrinsics_csv"] = Path(args.right_intrinsics_csv)
    if args.preview_mode:
        settings["preview_mode"] = args.preview_mode
    if args.preview_delay_ms is not None:
        settings["preview_delay_ms"] = args.preview_delay_ms

    settings["left_dir"] = project_path(settings["left_dir"])
    settings["right_dir"] = project_path(settings["right_dir"])
    if settings["left_intrinsics_csv"] != Path(""):
        settings["left_intrinsics_csv"] = project_path(settings["left_intrinsics_csv"])
    if settings["right_intrinsics_csv"] != Path(""):
        settings["right_intrinsics_csv"] = project_path(settings["right_intrinsics_csv"])
    output_path = project_path(Path(args.output) if args.output else settings["output"])
    if settings["board_type"] != "charuco":
        raise ValueError("This Python calibrator currently supports board_type=charuco only")
    if settings["cols"] <= 0 or settings["rows"] <= 0:
        raise ValueError("cols and rows must be positive")
    if not 0 < settings["marker_size_m"] < settings["square_size_m"]:
        raise ValueError("marker_size_m must be positive and smaller than square_size_m")
    if settings["preview_mode"] not in {"off", "key", "time"}:
        raise ValueError("preview_mode must be off, key or time")
    if settings["preview_delay_ms"] <= 0:
        raise ValueError("preview_delay_ms must be positive")
    has_left_intrinsics = settings["left_intrinsics_csv"] != Path("")
    has_right_intrinsics = settings["right_intrinsics_csv"] != Path("")
    if has_left_intrinsics != has_right_intrinsics:
        raise ValueError("Provide both left and right intrinsic CSV files, or neither")
    if has_left_intrinsics:
        left_k, left_d = load_intrinsics_csv(settings["left_intrinsics_csv"])
        right_k, right_d = load_intrinsics_csv(settings["right_intrinsics_csv"])
    else:
        left_k = left_d = right_k = right_d = None

    dictionary = dictionary_from_name(settings["aruco_dictionary"])
    board = create_charuco_board(
        int(settings["cols"]), int(settings["rows"]),
        float(settings["square_size_m"]), float(settings["marker_size_m"]), dictionary
    )
    detector = cv2.aruco.CharucoDetector(board) if hasattr(cv2.aruco, "CharucoDetector") else None
    left_paths, right_paths = paired_image_files(settings["left_dir"], settings["right_dir"])

    accepted_pairs = {}
    image_size = None
    automatic_rejections = set()
    deleted_pairs = set()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    if hasattr(board, "getChessboardCorners"):
        board_points = board.getChessboardCorners()
    else:
        board_points = board.chessboardCorners

    if settings["preview_mode"] != "off":
        cv2.namedWindow(CALIBRATION_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CALIBRATION_WINDOW, 1800, 950)

    pair_index = 0
    while pair_index < len(left_paths):
        left_path = left_paths[pair_index]
        right_path = right_paths[pair_index]
        display_index = pair_index + 1
        if not left_path.exists() or not right_path.exists():
            pair_index += 1
            continue
        left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None or left.shape != right.shape:
            print(f"Rejected pair {display_index}: unreadable or different image size")
            automatic_rejections.add(pair_index)
            pair_index += 1
            continue
        current_size = (left.shape[1], left.shape[0])
        if image_size is None:
            image_size = current_size
        elif image_size != current_size:
            raise ValueError("All calibration images must have the same resolution")

        corners_left, ids_left = detect_charuco(detector, board, left, dictionary)
        corners_right, ids_right = detect_charuco(detector, board, right, dictionary)
        found_left = ids_left is not None and len(ids_left) >= 4
        found_right = ids_right is not None and len(ids_right) >= 4
        if found_left:
            corners_left = cv2.cornerSubPix(
                left, corners_left.astype(np.float32), (5, 5), (-1, -1), criteria
            )
        if found_right:
            corners_right = cv2.cornerSubPix(
                right, corners_right.astype(np.float32), (5, 5), (-1, -1), criteria
            )

        common_ids = []
        pair_object_points = None
        pair_left_points = None
        pair_right_points = None
        if found_left and found_right:
            left_lookup = {
                int(corner_id): point
                for corner_id, point in zip(ids_left.reshape(-1), corners_left)
            }
            right_lookup = {
                int(corner_id): point
                for corner_id, point in zip(ids_right.reshape(-1), corners_right)
            }
            common_ids = sorted(left_lookup.keys() & right_lookup.keys())
            if len(common_ids) >= 6:
                pair_object_points = np.asarray(
                    [board_points[corner_id] for corner_id in common_ids], np.float32
                )
                pair_left_points = np.asarray(
                    [left_lookup[corner_id] for corner_id in common_ids], np.float32
                ).reshape(-1, 1, 2)
                pair_right_points = np.asarray(
                    [right_lookup[corner_id] for corner_id in common_ids], np.float32
                ).reshape(-1, 1, 2)

        pair_acceptable = pair_object_points is not None
        action = "keep"
        if settings["preview_mode"] != "off":
            action = calibration_preview(
                left,
                right,
                corners_left,
                corners_right,
                ids_left,
                ids_right,
                len(common_ids),
                pair_acceptable,
                left_path.name,
                right_path.name,
                display_index,
                len(left_paths),
                settings["preview_mode"],
                settings["preview_delay_ms"],
            )
        if action == "abort":
            cv2.destroyWindow(CALIBRATION_WINDOW)
            raise RuntimeError("Calibration review aborted; stereo.yml was not written")
        if action == "back":
            previous_index = pair_index - 1
            while previous_index >= 0 and (
                not left_paths[previous_index].exists()
                or not right_paths[previous_index].exists()
            ):
                previous_index -= 1
            if previous_index < 0:
                print("Already at the first available image pair")
            else:
                accepted_pairs.pop(previous_index, None)
                pair_index = previous_index
            continue
        if action == "delete":
            accepted_pairs.pop(pair_index, None)
            try:
                left_path.unlink()
                right_path.unlink()
            except OSError as error:
                raise RuntimeError(
                    f"Failed to delete pair {display_index}; inspect both paths: "
                    f"{left_path}, {right_path}"
                ) from error
            automatic_rejections.discard(pair_index)
            deleted_pairs.add(pair_index)
            print(
                f"Deleted pair {display_index}: {left_path.name}, {right_path.name}"
            )
            pair_index += 1
            continue
        if not pair_acceptable:
            print(
                f"Rejected pair {display_index}: left={0 if ids_left is None else len(ids_left)} "
                f"right={0 if ids_right is None else len(ids_right)} common={len(common_ids)}"
            )
            automatic_rejections.add(pair_index)
            pair_index += 1
            continue
        automatic_rejections.discard(pair_index)
        accepted_pairs[pair_index] = (
            pair_object_points,
            pair_left_points,
            pair_right_points,
        )
        pair_index += 1

    if settings["preview_mode"] != "off":
        cv2.destroyWindow(CALIBRATION_WINDOW)

    ordered_pairs = [accepted_pairs[index] for index in sorted(accepted_pairs)]
    object_points = [pair[0] for pair in ordered_pairs]
    image_points_left = [pair[1] for pair in ordered_pairs]
    image_points_right = [pair[2] for pair in ordered_pairs]

    if image_size is None or len(object_points) < 10:
        raise ValueError(f"Only {len(object_points)} valid pairs; at least 10 are required")

    if left_k is None:
        rms_left, left_k, left_d, _, _ = cv2.calibrateCamera(object_points, image_points_left, image_size, None, None)
        rms_right, right_k, right_d, _, _ = cv2.calibrateCamera(object_points, image_points_right, image_size, None, None)
    else:
        rms_left = float("nan")
        rms_right = float("nan")
    rms_stereo, left_k, left_d, right_k, right_d, rotation, translation, _, _ = cv2.stereoCalibrate(
        object_points, image_points_left, image_points_right,
        left_k, left_d, right_k, right_d, image_size,
        criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    storage = cv2.FileStorage(str(output_path), cv2.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        raise RuntimeError(f"Cannot create output: {output_path}")
    try:
        storage.write("image_width", image_size[0])
        storage.write("image_height", image_size[1])
        storage.write("camera_matrix_left", left_k)
        storage.write("distortion_left", left_d)
        storage.write("camera_matrix_right", right_k)
        storage.write("distortion_right", right_d)
        storage.write("rotation_left_to_right", rotation)
        storage.write("translation_left_to_right", translation)
        if np.isfinite(rms_left):
            storage.write("rms_left", float(rms_left))
        if np.isfinite(rms_right):
            storage.write("rms_right", float(rms_right))
        storage.write("rms_stereo", float(rms_stereo))
        storage.write("valid_pair_count", len(object_points))
    finally:
        storage.release()

    print(f"Calibration written to {output_path}")
    print(f"baseline_m={np.linalg.norm(translation):.6f}")
    print(
        f"valid_pairs={len(object_points)} auto_rejected={len(automatic_rejections)} "
        f"deleted_pairs={len(deleted_pairs)} rms_left={rms_left:.6f} "
        f"rms_right={rms_right:.6f} rms_stereo={rms_stereo:.6f}"
    )
    if rms_stereo > 1.0:
        print("Warning: stereo RMS exceeds 1 pixel; inspect pairs and board setup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
