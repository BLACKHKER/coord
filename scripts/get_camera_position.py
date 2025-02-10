"""
@Author  ：Zhao
@Date    ：2024/1/8 18:59
@File    ：get_camera_position.py
@Description: 获取相机的世界坐标(World Position)
@Version 1.0
"""

import numpy as np

# 输入的旋转矩阵 R 和平移向量 t
R = np.array([
    [9.998816e-01, 1.506446e-02, -3.128223e-03],
    [2.266580e-03, 5.687818e-02, 9.983786e-01],
    [1.521796e-02, -9.982675e-01, 5.683730e-02]
])

t = np.array([
    [-9.257329e+01],
    [6.754552e+01],
    [4.246720e+02]
])

Ow = -np.dot(R.T, t)

print("相机原点在世界坐标系下的坐标:", Ow)
