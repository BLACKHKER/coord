import os
import numpy as np
import cv2


class Camera:
    def __init__(self, filename: str = None) -> None:
        self.IntrinsicMatrix = None
        self.ExtrinsicsMatrix = None
        self.Distortion = None
        self.Matrix = None
        self.rvec = None
        self.tvec = None
        self.rotM = None
        self._flag = False
        if filename:
            self.loadFromCSV(filename)

    def loadFromCSV(self, filename: str) -> None:
        if not os.path.exists(filename):
            print("file not found")

        with open(filename, encoding='utf-8') as f:
            params = np.loadtxt(f, delimiter=',')
            n = params.shape[0]
            if n >= 5:
                self.IntrinsicMatrix = params[0:3, 0:3].copy()
                tmp1 = np.zeros(5)
                tmp1[0:2] = params[3, 0:2]
                tmp1[2:4] = params[4, 0:2]
                tmp1[4] = params[3, 2]
                self.Distortion = tmp1

                if n >= 9:
                    tmp2 = np.zeros((4, 4))
                    tmp2[0:3, 0:3] = params[5:8, 0:3].copy()
                    tmp2[0:3, 3] = params[8].copy()
                    tmp2[3, 3] = 1
                    self.ExtrinsicsMatrix = tmp2.copy()
                    self.rotM = params[5:8, 0:3].copy()
                    self.rvec = np.reshape(cv2.Rodrigues(self.rotM)[0], 3)
                    self.tvec = params[8].copy()

                    tmp1 = np.zeros((3, 4))
                    tmp1[0:3, 0:3] = self.IntrinsicMatrix.copy()
                    self.Matrix = np.matmul(tmp1, self.ExtrinsicsMatrix)
                    self._flag = True

    def saveMatric(self, filename: str):
        res = np.zeros((9, 3))
        res[0:3] = self.IntrinsicMatrix
        res[3, 0:2] = self.Distortion[0:2]
        res[4, 0:2] = self.Distortion[2:4]
        res[3, 2] = self.Distortion[4]
        res[5:8] = self.ExtrinsicsMatrix[0:3, 0:3]
        res[8] = self.ExtrinsicsMatrix[0:3, 3]
        np.savetxt(filename, res, delimiter=',')

    def transform(self, world_coord):
        """变换世界坐标到图像坐标

        :param world_coord:
        :return:
        """
        if not self._flag:
            print('相机参数未标定')
            return
        img_coord = cv2.projectPoints(world_coord, self.rvec, self.tvec, self.IntrinsicMatrix, self.Distortion)
        x = img_coord[0][0][0][0]
        y = img_coord[0][0][0][1]
        return round(x), round(y)


    def caliExtrinsicsMatrix(self, world_points, img_points):
        found, rvec, tvec = cv2.solvePnP(world_points, img_points, self.IntrinsicMatrix, self.Distortion)
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
        self.ExtrinsicsMatrix = tmp.copy()
        tmp1 = np.zeros((3, 4))
        tmp1[0:3, 0:3] = self.IntrinsicMatrix.copy()
        self.Matrix = np.matmul(tmp1, self.ExtrinsicsMatrix)
        self._flag = True
