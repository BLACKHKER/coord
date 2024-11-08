import glob
import cv2
import pickle
import numpy as np

# 图像坐标点(2D)
img_points = []
# 世界坐标点(3D)
obj_points = []

# 定义棋盘格的维度(高, 宽) 我的棋盘格是8 * 12，需要-1；
# 因为没法检测最外层角点，所以最外层一圈的棋盘格不参与计算。
CHECKER_BOARD = (7, 11)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 初始化为三维数组(x, y, z)，数据类型是float
obj = np.zeros((1, CHECKER_BOARD[0] * CHECKER_BOARD[1], 3), np.float32)
print(obj, end="\n")

# 根据棋盘格维度，创建二维数组
obj[0, :, :2] = np.mgrid[0:CHECKER_BOARD[0], 0:CHECKER_BOARD[1]].T.reshape(-1, 2)
print(17 * "=", "开始", 17 * "=", end="\n")
print(np.mgrid[0:CHECKER_BOARD[0], 0:CHECKER_BOARD[1]], end="\n")
# print(17 * "=", "转置", 17 * "=", end="\n")
# print(np.mgrid[0:CHECKER_BOARD[0], 0:CHECKER_BOARD[1]].T, end="\n")
# print(17 * "=", "置换为二列", 17 * "=", end="\n")
# print(np.mgrid[0:CHECKER_BOARD[0], 0:CHECKER_BOARD[1]].T.reshape(-1, 2), end="\n")
prev_img_shape = None
images = glob.glob('./ch_camera_20mm/*.jpg')

for path in images:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKER_BOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    # 如果找到角点
    if ret is True:
        print("找到角点" + "/n")
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners2)
        obj_points.append(obj)

        # 画图并展示角点
        img = cv2.drawChessboardCorners(img, CHECKER_BOARD, corners2, ret)

    desired_width = 640
    desired_height = 480
    img_resized = cv2.resize(img, (desired_width, desired_height))
    cv2.imshow('Visualize IMG', img_resized)
    cv2.waitKey(100)
cv2.destroyAllWindows()

# 校准摄像头
if len(obj_points) > 0 and len(img_points) > 0:
    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )

# 重投影误差，越低精确度越高
print("标定重投影误差 : \n", retval, end="\n")
# 相机内参矩阵
print("相机内参: \n", camera_matrix, end="\n")
# 畸变系数(k1,k2,p1,p2,k3)
print("畸变系数: \n", dist_coeffs, end="\n")
# 旋转向量
print("rvecs : \n", rvecs, end="\n")
# 位移向量
print("tvecs : \n", tvecs, end="\n")
