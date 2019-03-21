import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


"""
    降噪
    二值化 OTSU
    倾斜校正 霍夫变换
    字符切分
    识别
"""


def read_image(file):
    image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    return image


def pre_process(image):
    pass
    return np.array(image)


# 二值化
def binary_image(image):
    # 原图黑字白底
    threshold, ret = cv.threshold(image, 0, 255, cv.THRESH_OTSU)
    # plt.imshow(ret)
    # plt.show()
    return ret


# 获取图片有效区间
def get_valid_range(image, mode='vertical'):
    min_range = 2
    rows, cols = image.shape
    length = 0
    stat = []
    if mode == 'vertical':
        length = rows
        stat = np.zeros(length, dtype=int)
        for i in range(rows):
            for j in range(cols):
                if image[i][j] == 0:
                    stat[i] += 1
    elif mode == 'horizon':
        length = cols
        stat = np.zeros(length, dtype=int)
        for i in range(rows):
            for j in range(cols):
                if image[i][j] == 0:
                    stat[j] += 1

    valid_range = []
    threshold = 0
    can_begin = True
    start = 0
    for i in range(length):
        if can_begin and stat[i] > threshold:
            start = i
            can_begin = False
        if not can_begin and stat[i] <= threshold:
            if i - start >= min_range:
                valid_range.append((start, i))  # [start, end)
            can_begin = True
    print(valid_range)
    valid_image = []
    if mode == 'vertical':
        for item in valid_range:
            valid_image.append(image[item[0]:item[1]])
    elif mode == 'horizon':
        for item in valid_range:
            valid_image.append(image[:, item[0]:item[1]])
    return valid_image


def main():
    image = read_image("image/test4.jpg")
    image = pre_process(image)
    image = binary_image(image)
    valid_vertical_images = get_valid_range(image, 'vertical')
    valid_images = []
    for item in valid_vertical_images:
        img = get_valid_range(item, 'horizon')
        valid_images.extend(img)
    print(valid_images)
    for image in valid_images:
        height, width = image.shape
        print(image.shape, width/height)
        cv.imshow('g', image)
        cv.waitKey()


if __name__ == '__main__':
    main()
