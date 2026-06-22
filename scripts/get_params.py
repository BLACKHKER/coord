import numpy as np

from scripts.camera import Camera

W, H = 1200, 1200
world_points = np.array([
    [0,     0,    0],
    [W,     0,    0],
    [W,     H,    0],
    [0,     H,    0],
    [W/2,   0,    0],   # 下边中点
    [W,     H/2,  0],
    [W/2,   H,    0],
    [0,     H/2,  0],
    [W/2,   H/2,  0],   # 中心
], dtype=np.double)

# W, H = 1200, 2400
# world_points = np.array([
#     [0,     2400,    0],
#     [1200,     2400,    0],
#     [1200,     3600,    0],
#     [0,     3600,    0],
#     [W/2,   0,    0],   # 下边中点
#     [W,     H/2,  0],
#     [W/2,   H,    0],
#     [0,     H/2,  0],
#     [W/2,   H/2,  0],   # 中心
# ], dtype=np.double)

# 图像分辨率
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


def pattern_coverage(pixel_points, img_w: int, img_h: int) -> float:
    """计算外参标定图案(四边形)占图像面积的比例

    计算实际多边形面积

    Args:
        pixel_points: shape (4, 2) 的像素坐标数组，顶点顺序需为顺/逆时针
        img_w: 图像宽度(像素)
        img_h: 图像高度(像素)
    Returns:
        覆盖比例 0.0 ~ 1.0
    """
    pts = np.array(pixel_points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    # Shoelace 公式
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area / (img_w * img_h)


if __name__ == "__main__":
    # TODO csv路径
    params_path = "../csv/OpenCV_Camera_Intrinsics.csv"  # 焦距、畸变系数(内参)
    xy_path = "../csv/xy.csv"  # 像素坐标(外参)
    res_path = "../csv/world_params.csv"  # 内外参(整合)

    # 世界坐标系下坐标转换相机坐标系下坐标，提取像素坐标数据
    with open(xy_path, encoding="utf-8") as f:
        data = np.loadtxt(f, delimiter=",")
    img_points = data.reshape(-1, 2)  # 支持任意行数
    n = len(img_points)
    print(f"读取像素坐标：{n} 个点")

    ratio = pattern_coverage(img_points[:4], IMAGE_WIDTH, IMAGE_HEIGHT)
    print(f"标定图案占图像面积：{ratio * 100:.1f}%")

    camera = Camera()
    camera.load_intrinsics_matrix(params_path)  # 内参读取
    camera.solve_extrinsics_matrix(world_points[:n], img_points)
    camera.save_camera_params(res_path)
