# -*- coding: utf-8 -*-
import cv2
from skimage.measure import label
import numpy as np


def largest_connect_component(bw_img):
    labeled_img, num = label(bw_img, neighbors=4, background=0, return_num=True)

    max_label = 0
    max_num = 0
    for i in range(1, num+1):  # 这里从1开始，防止将背景设置为最大连通域
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)

    idx = np.where(lcc == True)
    min_x, max_x, min_y, max_y = np.min(idx[1]), np.max(idx[1]), np.min(idx[0]), np.max(idx[0])
    return min_x, max_x, min_y, max_y


if __name__ == "__main__":
    image = cv2.imread('./testmasks.jpg')
    image = cv2.resize(image, (224, 224))
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


    ret, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
    cv2.imshow("binary", binary_image)
    cv2.waitKey(0)

    min_x, max_x, min_y, max_y = largest_connect_component(binary_image)

    res = cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)
    cv2.imshow("res", res)
    cv2.waitKey(0)
