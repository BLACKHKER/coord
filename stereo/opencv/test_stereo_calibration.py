"""Run without cameras: python -m unittest discover -s stereo/opencv -v."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

import capture_stereo_pairs as capture
import stereo_calibrate as calibrate


class StereoCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.left = cls.root / "left"
        cls.right = cls.root / "right"
        cls.left.mkdir()
        cls.right.mkdir()
        cls.camera = np.array([[1000., 0., 640.], [0., 1000., 480.], [0., 0., 1.]])
        cls.translation = np.array([[-0.12], [0.002], [0.001]])
        cls.rotation = cv2.Rodrigues(np.array([0.005, -0.015, 0.003]))[0]
        cls.intrinsics = cls.root / "intrinsics.csv"
        values = np.zeros((5, 3))
        values[:3] = cls.camera
        np.savetxt(str(cls.intrinsics), values, delimiter=",", fmt="%.10f")

        dictionary = calibrate.dictionary_from_name("DICT_6X6_50")
        board = calibrate.create_charuco_board(10, 7, 0.068, 0.0535, dictionary)
        texture = board.generateImage((1100, 800))
        texture_corners = np.float32([[0, 0], [1099, 0], [1099, 799], [0, 799]])
        physical_corners = np.float32([[0, 0, 0], [0.748, 0, 0], [0.748, 0.544, 0], [0, 0.544, 0]])
        rng = np.random.default_rng(42)
        for index in range(14):
            rvec = rng.uniform(-0.35, 0.35, (3, 1))
            rotation = cv2.Rodrigues(rvec)[0]
            translation = np.array([
                [-0.35 + rng.uniform(-0.12, 0.12)],
                [-0.25 + rng.uniform(-0.10, 0.10)],
                [rng.uniform(1.5, 2.2)],
            ])
            right_rvec = cv2.Rodrigues(cls.rotation @ rotation)[0]
            right_translation = cls.rotation @ translation + cls.translation
            for directory, rv, tv in (
                (cls.left, rvec, translation),
                (cls.right, right_rvec, right_translation),
            ):
                projected = cv2.projectPoints(physical_corners, rv, tv, cls.camera, None)[0]
                homography = cv2.getPerspectiveTransform(texture_corners, projected.reshape(4, 2))
                image = cv2.warpPerspective(texture, homography, (1280, 960), borderValue=230)
                if not cv2.imwrite(str(directory / f"{index:06d}.png"), image):
                    raise RuntimeError("Could not write synthetic calibration image")

    def config(self, name, fixed=True):
        path = self.root / f"{name}.yml"
        output = self.root / f"{name}_stereo.yml"
        intrinsic_path = self.intrinsics.as_posix() if fixed else ""
        path.write_text(
            '%YAML:1.0\n'
            f'left_dir: "{self.left.as_posix()}"\n'
            f'right_dir: "{self.right.as_posix()}"\n'
            f'output: "{output.as_posix()}"\n'
            f'left_intrinsics_csv: "{intrinsic_path}"\n'
            f'right_intrinsics_csv: "{intrinsic_path}"\n'
            'board_type: charuco\ncols: 10\nrows: 7\n'
            'square_size_m: 0.068\nmarker_size_m: 0.0535\n'
            'aruco_dictionary: DICT_6X6_50\npreview_mode: off\n',
            encoding="utf-8",
        )
        return path, output

    def run_calibration(self, name, fixed):
        config, output = self.config(name, fixed)
        # An unrelated working directory verifies that absolute config paths work.
        result = subprocess.run(
            [sys.executable, str(Path(calibrate.__file__).resolve()), "--config", str(config)],
            cwd=str(self.root), capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        storage = cv2.FileStorage(str(output), cv2.FILE_STORAGE_READ)
        try:
            self.assertTrue(storage.isOpened())
            self.assertEqual(storage.getNode("image_width").real(), 1280)
            self.assertEqual(storage.getNode("image_height").real(), 960)
            self.assertEqual(storage.getNode("valid_pair_count").real(), 14)
            self.assertLess(storage.getNode("rms_stereo").real(), 1.0)
            rotation = storage.getNode("rotation_left_to_right").mat()
            translation = storage.getNode("translation_left_to_right").mat()
            np.testing.assert_allclose(rotation, self.rotation, atol=0.01)
            np.testing.assert_allclose(translation, self.translation, atol=0.015)
            for side in ("left", "right"):
                matrix = storage.getNode(f"camera_matrix_{side}").mat()
                distortion = storage.getNode(f"distortion_{side}").mat()
                self.assertTrue(np.isfinite(distortion).all())
                if fixed:
                    np.testing.assert_array_equal(matrix, self.camera)
                else:
                    np.testing.assert_allclose(matrix, self.camera, atol=25)
        finally:
            storage.release()

    def test_fixed_intrinsics_end_to_end(self):
        self.run_calibration("fixed", True)

    def test_estimated_intrinsics_end_to_end(self):
        self.run_calibration("estimated", False)

    def test_intrinsics_distortion_order_and_matlab_transpose(self):
        values = np.zeros((5, 3))
        values[:3] = self.camera
        values[3] = [0.1, -0.2, 0.3]
        values[4, :2] = [0.004, -0.005]
        for name in ("opencv.csv", "MATLAB.csv"):
            path = self.root / name
            export = values.copy()
            if name == "MATLAB.csv":
                export[:3] = export[:3].T
            np.savetxt(str(path), export, delimiter=",")
            matrix, distortion = calibrate.load_intrinsics_csv(path)
            np.testing.assert_array_equal(matrix, self.camera)
            np.testing.assert_array_equal(distortion, [[0.1, -0.2, 0.004, -0.005, 0.3]])

    def test_equal_counts_with_wrong_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            other = Path(directory)
            for index in range(14):
                (other / f"wrong_{index:06d}.png").touch()
            with self.assertRaisesRegex(ValueError, "filename sets"):
                calibrate.paired_image_files(self.left, other)

    def test_capture_resumes_and_rejects_unpaired_files(self):
        self.assertEqual(capture.next_index(self.left, self.right), 14)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "not paired"):
                capture.next_index(self.left, Path(directory))

    def test_abort_preserves_existing_output(self):
        config, output = self.config("abort")
        output.write_text("existing calibration", encoding="utf-8")
        with mock.patch.object(sys, "argv", ["stereo_calibrate.py", "--config", str(config), "--preview-mode", "key"]), \
             mock.patch.object(cv2, "namedWindow"), \
             mock.patch.object(cv2, "resizeWindow"), \
             mock.patch.object(cv2, "destroyWindow"), \
             mock.patch.object(calibrate, "calibration_preview", return_value="abort"):
            with self.assertRaisesRegex(RuntimeError, "aborted"):
                calibrate.main()
        self.assertEqual(output.read_text(encoding="utf-8"), "existing calibration")

    def test_default_paths_are_relative_to_coord(self):
        self.assertEqual(calibrate.PROJECT_ROOT, Path(__file__).resolve().parents[2])
        self.assertEqual(capture.PROJECT_ROOT, calibrate.PROJECT_ROOT)
        settings = calibrate.read_config(calibrate.DEFAULT_CONFIG)
        self.assertEqual(calibrate.project_path(settings["output"]), calibrate.PROJECT_ROOT / "stereo.yml")


if __name__ == "__main__":
    unittest.main()
