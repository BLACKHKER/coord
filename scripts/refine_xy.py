"""
@Author  ：Zhao
@File    ：refine_xy.py
@Description: 对 xy.csv 中的手动像素坐标进行亚像素精化
              使用 cornerSubPix：在初始点附近迭代求解梯度垂直条件，收敛后输出精确坐标
"""

import cv2
import numpy as np

# 外参标定图像路径
IMAGE_PATH = "../image/test2.jpg"

# 输入像素坐标
INPUT_XY = "../csv/xy_test2.csv"
# 输出梯度估计后的像素坐标
OUTPUT_XY = "../csv/xy_fix.csv"

# 搜索窗口半径，实际窗口为(2*WIN+1)×(2*WIN+1)，即默认11×11
WIN = 5

# 收敛条件：最大迭代次数 最小移动量(像素)
MAX_ITER = 100
EPSILON = 0.001

# 只精化前 N_CORNERS 个点（必须是真实角点才有效）
# 边中点和中心点是胶带边缘/内部，梯度方向单一，不适合 cornerSubPix，保持手动值
N_CORNERS = 4


def refine_xy(image_path: str, input_csv: str, output_csv: str,
              win: int, max_iter: int, epsilon: float, n_corners: int):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"图像不存在：{image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rough = np.loadtxt(input_csv, delimiter=",", dtype=np.float32)  # (N, 2)
    n_total = len(rough)
    n_refine = min(n_corners, n_total)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    corners = rough[:n_refine].reshape(-1, 1, 2)
    refined_corners = cv2.cornerSubPix(gray, corners, (win, win), (-1, -1), criteria)
    refined_corners = refined_corners.reshape(-1, 2)

    # 精化点 + 未精化点合并
    refined = np.vstack([refined_corners, rough[n_refine:]]) if n_refine < n_total else refined_corners

    print(f"窗口大小：{2 * win + 1} × {2 * win + 1}  收敛阈值：{epsilon}px  最大迭代：{max_iter}")
    print(f"精化前 {n_refine} 个角点，其余 {n_total - n_refine} 个点保持手动值不变")
    print(f"{'点':>4}  {'粗略坐标':^24}  {'精化坐标':^24}  {'偏移':>10}")
    for i, (r, f) in enumerate(zip(rough, refined)):
        dx, dy = f[0] - r[0], f[1] - r[1]
        shift = np.hypot(dx, dy)
        tag = "" if i < n_refine else "  (跳过)"
        print(f"  #{i + 1}  ({r[0]:8.2f}, {r[1]:8.2f})  →  ({f[0]:8.4f}, {f[1]:8.4f})  {shift:+.4f}px{tag}")

    np.savetxt(output_csv, refined, delimiter=",", fmt="%.4f")
    print(f"\n已保存至：{output_csv}")

    # 可视化：红色=粗略点，绿色=精化点
    preview = img.copy()
    for r, f in zip(rough, refined):
        cv2.circle(preview, (int(r[0]), int(r[1])), 6, (0, 0, 255), -1)
        cv2.circle(preview, (int(round(f[0])), int(round(f[1]))), 4, (0, 255, 0), -1)

    h, w = preview.shape[:2]
    scale = min(1280 / w, 960 / h, 1.0)
    if scale < 1.0:
        preview = cv2.resize(preview, (int(w * scale), int(h * scale)))

    cv2.imshow("Refine XY  (red=rough  green=refined)", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    refine_xy(IMAGE_PATH, INPUT_XY, OUTPUT_XY, WIN, MAX_ITER, EPSILON, N_CORNERS)
