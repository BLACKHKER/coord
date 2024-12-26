import numpy as np
from scripts.camera import Camera

square_points = np.array([[0, 0, 0], [0, 205, 0], [200, 205, 0], [200, 0, 0]], dtype=np.double)

if __name__ == "__main__":
    folder = 1
    xy_path = '../a/xy.csv'.format(folder)  # 保存的四个像素坐标，x、y
    params_path = '../a/params.csv'  # matlab计算后得到的params.csv内参
    res_path = '../a/world_params.csv'.format(folder)  # 计算外参后生成的world_params.csv文件，存储相机内外参数

    c1 = Camera(params_path)
    with open(xy_path, encoding='utf-8') as f:
        data = np.loadtxt(f, delimiter=',')
    img_points = data[0:4]
    c1.caliExtrinsicsMatrix(square_points, img_points)
    c1.saveMatric(res_path)
