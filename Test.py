"""
@Author  ：Zhao
@Date    ：2024/11/25 10:58
@File    ：Test.py
@Description: OpenCV内参标定方式，实现MATLAB的标定、导出CSV功能
@Version 1.0
"""

import glob
import cv2
import numpy
import pickle

# 不使用科学计数法e作为输出
numpy.set_printoptions(suppress=True)

# 图像坐标点(2D)
img_points = []
# 世界坐标点(3D)
obj_points = []

# 定义棋盘格的二维维度(高, 宽) 我的棋盘格是8 * 12，需要-1；
# 因为没法检测最外层角点，所以最外层一圈的棋盘格不参与计算。
CHECKER_BOARD = (8, 11)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 初始化为三维数组(x, y, z)，数据类型是float
obj = numpy.zeros((1, CHECKER_BOARD[0] * CHECKER_BOARD[1], 3), numpy.float32)

"""#根据棋盘格二维维度宽高，创建多维数组。

    1. mgrid[维度A, 维度B]
    参数个数决定矩阵维度，即两个参数表示仅生成行、列矩阵(x, y)。
    np.mgrid[x:test, y:b, z:c]，返回一个多维(三维)，x, y, z(行、列、深度)矩阵，其中每个维度大小，取决输入的切片区间。
    在这个例子中，有两个切片，返回的数组是一个包含两个元素(每个元素是一个二维数组)的多维数组。
    x:a表示切片区间，mgrid[0:4, 0:4]表示生成一个包含两个矩阵(4*4)的多维数组，分别对应行坐标矩阵和列坐标矩阵：
    [    
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3]
        ],
        [    
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3]
        ]
    ]
    以上实际为三维矩阵(2*3*3)，两个矩阵，每个矩阵三行三列；

    2. .T
    矩阵转置，二维是矩阵行列互换，三维是[D,R,C] -> [R,C,D]，Depth深度、Rows行、Column列。
    三维转置，原本两个矩阵[行矩阵, 列矩阵]的每个点会组合在一起，形成[行坐标, 列坐标]的对。
    深度在前面，表示两个矩阵，在后面表示矩阵中每个元素包含一个行列坐标的对(倒置坐标系)：
    [
        [
            [0, 0], [1, 0], [2, 0], [3, 0]
        ], 
        [
            [0, 1], [1, 1], [2, 1], [3, 1]
        ],
        [
            [0, 2], [1, 2], [2, 2], [3, 2]
        ],
        [
            [0, 3], [1, 3], [2, 3], [3, 3]
        ]
    ]

    3. reshape(x, y)
    数组展平，将原来的7*11*2的矩阵转换成
"""
obj[0, :, :2] = numpy.mgrid[0:CHECKER_BOARD[0], 0:CHECKER_BOARD[1]].T.reshape(-1, 2)
# print(17 * "=", "开始", 17 * "=", end="\n")
# print(numpy.mgrid[0:CHECKER_BOARD[0], 0:CHECKER_BOARD[1]], end="\n")
print(17 * "=", "转置", 17 * "=", end="\n")
print(numpy.mgrid[0:CHECKER_BOARD[0], 0:CHECKER_BOARD[1]].T, end="\n")
print(17 * "=", "全展平", 17 * "=", end="\n")
print(numpy.mgrid[0:CHECKER_BOARD[0], 0:CHECKER_BOARD[1]].T.reshape(-1, 2), end="\n")

images = glob.glob('./bj_camera_15mm/*.jpg')

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
    else:
        print(path)

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

# 重投影误差，越低精确度越高，1以内可接受
print("标定重投影误差 : \n", retval, end="\n")
# 相机内参矩阵
print("内参: \n", camera_matrix, end="\n")
# 径/切畸变系数(k1,k2,p1,p2,k3)
print("畸变: \n", dist_coeffs, end="\n")
# 旋转向量
# print("rvecs : \n", rvecs, end="\n")
# 位移向量
# print("tvecs : \n", tvecs, end="\n")

# 构建用于CSV导出的数组
# 创建5行3列的数组
csv_data = numpy.zeros((5, 3))
# 第1-3行是内参矩阵
csv_data[:3, :] = camera_matrix
# 第4行是径向畸变 (k1, k2, k3)
csv_data[3, :3] = dist_coeffs[0, :3]
# 第5行是切向畸变 (p1, p2)
csv_data[4, :2] = dist_coeffs[0, 3:5]
# 第5行的第三个值为0 (p3占位)
csv_data[4, 2] = 0
numpy.savetxt("opencv_csv/params.csv", csv_data, delimiter=",", fmt="%.10f")