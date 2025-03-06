"""
@Author  ：Zhao
@Date    ：2025/2/27 16:25
@File    ：test.py
@Description: TODO
@Version 1.0 
"""

import numpy as np

import numpy as np


def compute_camera_position(R, t):
    t = t.reshape(3, 1)
    camera_position = -np.dot(R.T, t)
    return camera_position.flatten()


R = np.array([
    [9.999840778066133584e-01, 4.163992758607578705e-03, 3.808582093565274045e-03],
    [-4.023979160885413943e-03, 5.300635421949230486e-02, 9.985860673993356418e-01],
    [3.956226101971716848e-03, -9.985854933738342876e-01, 5.302226606215641042e-02]
])

t = np.array([
    -8.983094649850388578e+02,
    7.033914177500561209e+02,
    4.436501106751919906e+03
])

camera_pos = compute_camera_position(R, t)
print(camera_pos)
