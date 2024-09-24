import numpy as np
from numpy.linalg import lstsq
import csv
import cv2
import os


def get_point(k, b):
    n = k.shape[0]
    one = np.ones(n)

    a = np.c_[one, -k]
    solu = lstsq(a, b, rcond=-1)

    y = solu[0][0]
    x = solu[0][1]
    return round(x), round(y)


if __name__ == "__main__":
    folder = 1
    img_path = '../data/image/area.jpg'
    kb_path = '../data/kb.csv'  # 前面拟合的kb(两条直线)，运行一次后自动删除
    xy_path = '../data/xy.csv'  # 保存两直线kb交点x,y的文件路径，自动生成/添加新标点

    data = []
    with open(kb_path, encoding="utf-8") as f:
        data = np.loadtxt(f, delimiter=',')

    k = data[:, 0]
    b = data[:, 1]
    x, y = get_point(k, b)

    img = cv2.imread(img_path)
    h, w, _ = img.shape
    s = 1
    cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
    while 1:
        cv2.namedWindow('res', 0)
        cv2.resizeWindow('res', int(w / s), int(h / s))
        cv2.imshow('res', img)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # 空格保存
            with open(xy_path, 'a', encoding='utf-8', newline="") as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow([x, y])
            os.remove(kb_path)  # 保存之后删除kb文件
            break
        if key == 27:  # ESC直接退出
            break
    cv2.destroyAllWindows()
