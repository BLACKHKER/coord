"""
@Author  ：Zhao
@Date    ：2025/2/11 15:29
@File    ：camera.py
@Description: 基于OpenCV 相机标定相关工具函数
@Version 1.0
"""

import os
import numpy as np
import cv2


class Camera:
    """
        相机类
    """

    def __init__(self):
        """相机对象初始化
        初始化提取内外参 该类所需的变量：
        intrinsic_matrix(ndarray): 内参矩阵
        distortion_coefficients(ndarray): 畸变系数
        ------
        extrinsic_matrix(ndarray): 外参矩阵
        rotation_matrix(ndarray): 旋转矩阵
        translation_vector(ndarray): 平移向量
        ------
        rotation_vector(ndarray): 旋转向量

        其中 畸变系数包含径向畸变(Radial Distortion)和切向畸变(Tangential Distortion)
        是一个3 * 2 的矩阵，为了与内参矩阵对齐，切向畸变补了一个0
        """
        self.intrinsic_matrix = None
        self.distortion_coefficients = None
        self.extrinsic_matrix = None
        self.rotation_matrix = None
        self.translation_vector = None
        self.rotation_vector = None
        self.flag = False
        self.matrix = None

    def load_intrinsics_matrix(self, file_path):
        """内参获取
        从CSV中获取相机的像素坐标、畸变系数等数据
        像素坐标包含水平、垂直焦距 fx, fy
        主点坐标是相机的光心点 cx, cy
        以及转换齐次坐标需要的数据 0, 0, 1

        Args:
            file_path(str): 内参文件路径，可以是MATLAB导出的，也可以是OpenCV导出的
        """

        if not os.path.exists(file_path):
            print("CSV路径错误、文件不存在")

        # 内参从params.csv读
        with open(file_path, encoding='utf-8') as f:
            intrinsic_params = np.loadtxt(f, delimiter=',')
            n = intrinsic_params.shape[0]  # 拿矩阵第一维度大小

            # 内参矩阵3、畸变系数2
            if n >= 5:
                # 内参矩阵
                if 'OpenCV' in file_path:
                    self.intrinsic_matrix = intrinsic_params[0:3, 0:3].copy()
                else:
                    self.intrinsic_matrix = intrinsic_params[0:3, 0:3].T.copy()

                # 畸变系数
                dist_coeffs = np.zeros(5)
                dist_coeffs[0:2] = intrinsic_params[3, 0:2]  # k1, k2
                dist_coeffs[2:4] = intrinsic_params[4, 0:2]  # p1, p2
                dist_coeffs[4] = intrinsic_params[3, 2]  # k3
                self.distortion_coefficients = dist_coeffs

                # if n >= 9:
                #     tmp2 = np.zeros((4, 4))
                #     tmp2[0:3, 0:3] = intrinsic_params[5:8, 0:3].copy()
                #     tmp2[0:3, 3] = intrinsic_params[8].copy()
                #     tmp2[3, 3] = 1
                #     self.extrinsic_matrix = tmp2.copy()
                #     self.rotM = intrinsic_params[5:8, 0:3].copy()
                #     self.rvec = np.reshape(cv2.Rodrigues(self.rotM)[0], 3)
                #     self.tvec = intrinsic_params[8].copy()
                #
                #     dist_arr = np.zeros((3, 4))
                #     dist_arr[0:3, 0:3] = self.intrinsic_matrix.copy()
                #
                #     self.Matrix = np.matmul(dist_arr, self.extrinsic_matrix)
                #     self._flag = True

    # def transform(self, world_coord):
    #     """变换世界坐标到图像坐标
    #
    #     """
    #     if not self._flag:
    #         print('相机参数未标定')
    #         return
    #     img_coord = cv2.projectPoints(world_coord, self.rvec, self.tvec, self.intrinsic_matrix, self.distortion)
    #     x = img_coord[0][0][0][0]
    #     y = img_coord[0][0][0][1]
    #     return round(x), round(y)

    def solve_extrinsics_matrix(self, world_points, img_points):
        found, rvec, tvec = cv2.solvePnP(world_points, img_points, self.intrinsic_matrix, self.distortion_coefficients)

        if not found:
            print('无法标定外参矩阵')
            return

        self.rotation_vector = rvec
        self.translation_vector = tvec
        self.rotation_matrix = cv2.Rodrigues(rvec)[0]

        # 外参矩阵
        matrix = np.zeros((4, 4))
        matrix[0:3, 0:3] = self.rotation_matrix.copy()
        matrix[0:3, 3] = tvec.reshape(3, ).copy()  # 行列互换
        matrix[3, 3] = 1  # 凑齐次矩阵3 * 3用的
        self.extrinsic_matrix = matrix.copy()

        # 投影矩阵
        tmp = np.zeros((3, 4))
        tmp[0:3, 0:3] = self.intrinsic_matrix.copy()
        self.matrix = np.matmul(tmp, self.extrinsic_matrix)
        self.flag = True

    def save_camera_params(self, file_path: str):
        matrix_params = np.zeros((9, 3))
        matrix_params[0:3] = self.intrinsic_matrix
        matrix_params[3, 0:2] = self.distortion_coefficients[0:2]
        matrix_params[4, 0:2] = self.distortion_coefficients[2:4]
        matrix_params[3, 2] = self.distortion_coefficients[4]
        matrix_params[5:8] = self.extrinsic_matrix[0:3, 0:3]
        matrix_params[8] = self.extrinsic_matrix[0:3, 3]
        np.savetxt(file_path, matrix_params, delimiter=',')
