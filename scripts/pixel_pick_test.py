"""
@Author  ：Zhao
@File    ：pixel_pick_test.py
@Description: 像素坐标转世界坐标 手动测试工具
              模拟 YOLO 底边中点，通过鼠标点选像素点，反投影到地面平面(Z_w=0)
              得到相对外参原点的世界坐标
"""

import cv2
import numpy as np

# 测试图像
IMAGE_PATH = "../image/test2.jpg"
# 内外参
PARAMS_CSV = "../csv/world_params.csv"

# 预览窗口分辨率（不影响计算精度，仅用于显示）
PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 960


def load_params(csv_path: str):
    p = np.loadtxt(csv_path, delimiter=",")
    K = p[0:3, 0:3].copy()
    dist = np.array([p[3, 0], p[3, 1], p[4, 0], p[4, 1], p[3, 2]])  # k1,k2,p1,p2,k3
    R = p[5:8, 0:3].copy()
    t = p[8, 0:3].reshape(3, 1).copy()
    return K, dist, R, t


def pixel_to_world(u: float, v: float, K, dist, R, t):
    """像素坐标 (u,v) 反投影到世界地面平面 Z_w=0，返回 (X_w, Y_w) 单位与标定时一致"""
    # 去畸变，得到归一化相机坐标 (x_n, y_n)
    pt = np.array([[[u, v]]], dtype=np.float64)
    pt_undist = cv2.undistortPoints(pt, K, dist)
    x_n, y_n = pt_undist[0, 0]

    t_ = t.flatten()
    A = np.array([
        [R[0, 0] - R[2, 0] * x_n, R[0, 1] - R[2, 1] * x_n],
        [R[1, 0] - R[2, 0] * y_n, R[1, 1] - R[2, 1] * y_n],
    ])
    b = np.array([
        t_[2] * x_n - t_[0],
        t_[2] * y_n - t_[1],
    ])
    X_w, Y_w = np.linalg.solve(A, b)
    return X_w, Y_w


class PickTool:
    def __init__(self, image_path: str, K, dist, R, t):
        self.K = K
        self.dist = dist
        self.R = R
        self.t = t

        self.orig = cv2.imread(image_path)
        if self.orig is None:
            raise FileNotFoundError(f"图像不存在：{image_path}")

        self.orig_h, self.orig_w = self.orig.shape[:2]
        self.scale_x = self.orig_w / PREVIEW_WIDTH
        self.scale_y = self.orig_h / PREVIEW_HEIGHT

        # 显示画布（在缩放图上叠加标注）
        self.canvas = cv2.resize(self.orig, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
        self.base = self.canvas.copy()

        self.points = []  # [(u_orig, v_orig, X_w, Y_w), ...]

        cv2.namedWindow("Pixel Pick Test")
        cv2.setMouseCallback("Pixel Pick Test", self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # 将预览坐标映射回原始图像坐标
        u = x * self.scale_x
        v = y * self.scale_y

        try:
            X_w, Y_w = pixel_to_world(u, v, self.K, self.dist, self.R, self.t)
        except np.linalg.LinAlgError:
            print("坐标求解失败(奇异矩阵)，该点可能不在地面平面上")
            return

        self.points.append((u, v, X_w, Y_w))
        print(f"像素 ({u:.1f}, {v:.1f})转世界坐标 X={X_w:.1f}  Y={Y_w:.1f}")
        self._redraw()

    def _redraw(self):
        self.canvas = self.base.copy()
        for idx, (u, v, X_w, Y_w) in enumerate(self.points):
            # 映射回预览坐标绘制
            px = int(u / self.scale_x)
            py = int(v / self.scale_y)

            cv2.circle(self.canvas, (px, py), 6, (0, 0, 255), -1)
            cv2.circle(self.canvas, (px, py), 6, (255, 255, 255), 1)

            label = f"#{idx + 1}  X={X_w:.0f}  Y={Y_w:.0f}"
            # 文字背景
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(self.canvas, (px + 8, py - th - 4), (px + 8 + tw, py + 2), (0, 0, 0), -1)
            cv2.putText(self.canvas, label, (px + 8, py), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        self._draw_help()

    def _draw_help(self):
        lines = ["c 清除 s 打印汇总 q 退出"]
        cv2.putText(self.canvas, lines[0], (10, PREVIEW_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def run(self):
        self._draw_help()
        while True:
            cv2.imshow("Pixel Pick Test", self.canvas)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("c"):
                self.points.clear()
                self.canvas = self.base.copy()
                self._draw_help()
                print("已清除所有标注")
            elif key == ord("s"):
                for idx, (u, v, X_w, Y_w) in enumerate(self.points):
                    print(f"  #{idx + 1}  像素({u:.1f}, {v:.1f})  →  世界 X={X_w:.1f}  Y={Y_w:.1f}")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    K, dist, R, t = load_params(PARAMS_CSV)
    tool = PickTool(IMAGE_PATH, K, dist, R, t)
    tool.run()
