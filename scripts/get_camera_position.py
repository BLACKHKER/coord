"""
@Author  ：Zhao
@Date    ：2024/1/8 18:59
@File    ：get_camera_position.py
@Description: 获取相机的世界坐标(World Position)
@Version 1.0
"""

import numpy as np
import tkinter as tk
from tkinter import messagebox

# TODO 内外参文件位置
file_name = "world_params.csv"


def load_camera_params(file_name: str) -> str:
    """从world_params.csv中加载(内)外参

    """
    camera_params = np.loadtxt(file_name, delimiter=',')
    R = camera_params[5:8, :3]
    t = camera_params[8, :3].reshape(3, 1)

    return R, t


def compute_camera_position(R, t) -> str:
    """坐标转换

    """
    camera_position = -np.dot(R.T, t)

    return camera_position.flatten()


def calc_camera_origin():
    """计算相机原点在世界坐标系的坐标

    """
    try:
        # 获取输入的矩阵 R 和 t
        R_values = np.array([
            [float(entry_R11.get()), float(entry_R12.get()), float(entry_R13.get())],
            [float(entry_R21.get()), float(entry_R22.get()), float(entry_R23.get())],
            [float(entry_R31.get()), float(entry_R32.get()), float(entry_R33.get())]
        ])

        t_values = np.array([
            [float(entry_t1.get())],
            [float(entry_t2.get())],
            [float(entry_t3.get())]
        ])

        # 计算相机原点
        Ow = -np.dot(R_values.T, t_values.reshape(3, 1))
        result_label.config(text=f"相机原点在世界坐标系下的坐标: {Ow.flatten()}")

    except ValueError:
        messagebox.showerror("输入错误", "请输入有效的数字！")


# 创建主窗口
root = tk.Tk()
root.title("相机原点计算")

# 创建输入标签和输入框
tk.Label(root, text="旋转矩阵 R (3x3):").grid(row=0, column=0, columnspan=3)

tk.Label(root, text="R11:").grid(row=1, column=0)
entry_R11 = tk.Entry(root)
entry_R11.grid(row=1, column=1)

tk.Label(root, text="R12:").grid(row=2, column=0)
entry_R12 = tk.Entry(root)
entry_R12.grid(row=2, column=1)

tk.Label(root, text="R13:").grid(row=3, column=0)
entry_R13 = tk.Entry(root)
entry_R13.grid(row=3, column=1)

tk.Label(root, text="R21:").grid(row=1, column=2)
entry_R21 = tk.Entry(root)
entry_R21.grid(row=1, column=3)

tk.Label(root, text="R22:").grid(row=2, column=2)
entry_R22 = tk.Entry(root)
entry_R22.grid(row=2, column=3)

tk.Label(root, text="R23:").grid(row=3, column=2)
entry_R23 = tk.Entry(root)
entry_R23.grid(row=3, column=3)

tk.Label(root, text="R31:").grid(row=1, column=4)
entry_R31 = tk.Entry(root)
entry_R31.grid(row=1, column=5)

tk.Label(root, text="R32:").grid(row=2, column=4)
entry_R32 = tk.Entry(root)
entry_R32.grid(row=2, column=5)

tk.Label(root, text="R33:").grid(row=3, column=4)
entry_R33 = tk.Entry(root)
entry_R33.grid(row=3, column=5)

tk.Label(root, text="平移向量 t (3x1):").grid(row=4, column=0, columnspan=3)

tk.Label(root, text="t1:").grid(row=5, column=0)
entry_t1 = tk.Entry(root)
entry_t1.grid(row=5, column=1)

tk.Label(root, text="t2:").grid(row=6, column=0)
entry_t2 = tk.Entry(root)
entry_t2.grid(row=6, column=1)

tk.Label(root, text="t3:").grid(row=7, column=0)
entry_t3 = tk.Entry(root)
entry_t3.grid(row=7, column=1)

# 创建计算按钮
calc_button = tk.Button(root, text="来使计算", command=calc_camera_origin)
calc_button.grid(row=8, column=0, columnspan=6)

# 显示结果标签
result_label = tk.Label(root, text="相机原点在世界坐标系下的坐标: ")
result_label.grid(row=9, column=0, columnspan=6)

if __name__ == "__main__":
    R, t = load_camera_params("../csv/world_params.csv")
    print(compute_camera_position(R, t))

    # 窗口版本
    # root.mainloop()
