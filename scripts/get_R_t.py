"""
@Author  ：Zhao
@Date    ：2025/1/8 18:02
@File    ：get_R_t.py
@Description: FIXME (废弃)计算外参(多世界坐标）&像素坐标 废弃
@Version 1.0
"""

import cv2
import numpy as np
import csv


def calibrate_extrinsics(object_points, image_points, camera_matrix, dist_coeffs, output_csv='out_params.csv'):
    ret, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

    if not ret:
        raise ValueError("无法计算外参，check输入数据是否正确")

    # 将旋转向量转换为旋转矩阵
    r, _ = cv2.Rodrigues(rvec)

    # 保存到CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 保存旋转矩阵
        for row in r:
            writer.writerow([f"{x:e}" for x in row])

        # 保存平移向量
        writer.writerow([f"{x:e}" for x in tvec.flatten()])

    print(f"保存成功，外参、向量 {output_csv}")


# 世界坐标点
object_points = np.array([
    [0, 0, 0],
    [0, 200, 0],
    [200, 200, 0],
    [200, 0, 0],
    [100, 100, 0]
], dtype=np.double)

# 图像坐标点
image_points = np.array([
    [743, 777],
    [495, 1036],
    [1685, 1034],
    [1362, 777],
    [1040, 867]
], dtype=np.double)

# 内参(巨坑、MATLAB的矩阵需要转置才能给OpenCV)
camera_matrix = np.array([
    [1333.07404438552, 0, 1025.64747406552],
    [0, 1333.18501315671, 564.639881552654],
    [0, 0, 1]
], dtype=np.double)

# 畸变系数
dist_coeffs = np.array([
    0.0549661176311769,
    -0.123970076251731,
    0.0005130163793845,
    0.000745376583766039,
    0.137148776846108
])

# 调用函数计算外参并保存到CSV
calibrate_extrinsics(object_points, image_points, camera_matrix, dist_coeffs)
