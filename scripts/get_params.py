import numpy as np

from scripts.camera import Camera

# 世界坐标点(矩阵/非矩阵)，对应像素坐标，逆时针
world_points = np.array([[0, 0, 0], [74, 0, 0], [74, 74, 0], [0, 74, 0]], dtype=np.double)

if __name__ == "__main__":
    params_path = '../csv/params.csv'  # 焦距、畸变系数(内参)
    xy_path = '../csv/xy.csv'  # 像素坐标(外参)
    res_path = '../csv/world_params.csv'  # 内外参(整合)

    # 世界坐标系下坐标转换相机坐标系下坐标，提取像素坐标数据
    with open(xy_path, encoding='utf-8') as f:
        data = np.loadtxt(f, delimiter=',')
    img_points = data[0:4]

    camera = Camera()
    camera.load_from_csv(params_path)  # 内参读取
    camera.caliExtrinsicsMatrix(world_points, img_points)
    camera.saveMatric(res_path)
