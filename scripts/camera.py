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
        distortion(ndarray): 畸变系数
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
        self.distortion = None
        self.extrinsic_matrix = None
        self.rotation_matrix = None
        self.translation_vector = None
        self.rotation_vector = None
        self._flag = False

    def load_from_csv(self, file_path):
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
                self.intrinsic_matrix = intrinsic_params[0:3, 0:3].copy()  # 内参矩阵
                dist_arr = np.zeros(5)
                dist_arr[0:2] = intrinsic_params[3, 0:2]
                dist_arr[2:4] = intrinsic_params[4, 0:2]
                dist_arr[4] = intrinsic_params[3, 2]
                self.distortion = dist_arr

                if n >= 9:
                    tmp2 = np.zeros((4, 4))
                    tmp2[0:3, 0:3] = intrinsic_params[5:8, 0:3].copy()
                    tmp2[0:3, 3] = intrinsic_params[8].copy()
                    tmp2[3, 3] = 1
                    self.extrinsic_matrix = tmp2.copy()
                    self.rotM = intrinsic_params[5:8, 0:3].copy()
                    self.rvec = np.reshape(cv2.Rodrigues(self.rotM)[0], 3)
                    self.tvec = intrinsic_params[8].copy()

                    dist_arr = np.zeros((3, 4))
                    dist_arr[0:3, 0:3] = self.intrinsic_matrix.copy()

                    self.Matrix = np.matmul(dist_arr, self.extrinsic_matrix)
                    self._flag = True

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

    def caliExtrinsicsMatrix(self, world_points, img_points):
        found, rvec, tvec = cv2.solvePnP(world_points, img_points, self.intrinsic_matrix, self.distortion)
        print(type(found))
        if not found:
            print('无法标定外参矩阵')
            return
        self.rvec = rvec
        self.tvec = tvec
        self.rotM = cv2.Rodrigues(rvec)[0]
        tmp = np.zeros((4, 4))
        tmp[0:3, 0:3] = self.rotM.copy()
        tmp[0:3, 3] = tvec.reshape(3, ).copy()
        tmp[3, 3] = 1
        self.extrinsic_matrix = tmp.copy()
        tmp1 = np.zeros((3, 4))
        tmp1[0:3, 0:3] = self.intrinsic_matrix.copy()
        self.Matrix = np.matmul(tmp1, self.extrinsic_matrix)
        self._flag = True

    def saveMatric(self, file_path: str):
        res = np.zeros((9, 3))
        res[0:3] = self.intrinsic_matrix
        res[3, 0:2] = self.distortion[0:2]
        res[4, 0:2] = self.distortion[2:4]
        res[3, 2] = self.distortion[4]
        res[5:8] = self.extrinsic_matrix[0:3, 0:3]
        res[8] = self.extrinsic_matrix[0:3, 3]
        np.savetxt(file_path, res, delimiter=',')
