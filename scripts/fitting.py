import cv2
import numpy as np
from numpy.linalg import lstsq
import csv


def get_pixel(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param.append([x, y])


def fitting(x, y):
    n = x.shape[0]
    one = np.ones(n)
    a = np.c_[x, one]
    solu = lstsq(a, y, rcond=-1)
    k = solu[0][0]
    b = solu[0][1]
    return k, b


def draw_line(img, k, b):
    res = img.copy()
    h, w, _ = img.shape

    x1 = 0
    y1 = round(k * x1 + b)

    x2 = w
    y2 = round(k * x2 + b)

    y3 = 0
    x3 = round((y3 - b) / k)

    y4 = h
    x4 = round((y4 - b) / k)

    l = []

    if y1 >= 0:
        l.append((x1, y1))

    if y2 >= 0:
        l.append((x2, y2))
        if len(l) == 2:
            cv2.line(res, l[0], l[1], (255, 255, 151), thickness=1)
            return res

    if x3 >= 0:
        l.append((x3, y3))
        if len(l) == 2:
            cv2.line(res, l[0], l[1], (255, 255, 151), thickness=1)
            return res

    l.append((x4, y4))
    cv2.line(res, l[0], l[1], (255, 255, 151), thickness=1)
    return res


def main(filename: str, kb_path: str):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).copy()
    h, w, _ = img.shape

    l = []
    s = 1

    while 1:
        cv2.namedWindow('ScribeWindow', 0)
        cv2.resizeWindow('ScribeWindow', int(w / s), int(h / s))
        cv2.setMouseCallback('ScribeWindow', get_pixel, l)
        cv2.imshow('ScribeWindow', img)
        if len(l) == 4:
            break
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

    l = np.array(l)
    mask = np.zeros((h, w))
    cv2.drawContours(mask, [l], 0, (255, 255, 255), -1)
    posi = np.where(mask == 255)
    poi = np.zeros((h, w), dtype=np.uint8)
    poi[posi] = gray[posi]

    poi = cv2.Canny(poi, 50, 230, apertureSize=5)
    poi = cv2.drawContours(poi, [l], 0, (0, 0, 0), 7)
    y, x = np.where(poi != 0)
    k, b = fitting(x, y)
    res = draw_line(img, k, b)

    while 1:
        cv2.namedWindow('res', 0)
        cv2.resizeWindow('res', int(w / s), int(h / s))
        cv2.imshow('res', res)
        key = cv2.waitKey(1) & 0xFF

        # 空格保存
        if key == 32:
            with open(kb_path, 'a', encoding='utf-8', newline="") as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow([k, b])
            break

        # ESC退出
        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    folder = 1
    img_path = '../data/image/area.jpg' # 用来划线的图片的路径
    kb_path = '../data/kb.opencv_csv'   # 划线后生成的，用来保存直线参数的k\b文件路径
    main(img_path, kb_path)
    pass
