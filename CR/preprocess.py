import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as spi
from scipy.signal import find_peaks

import filter

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


# 统计二值化后两种灰度的数目
def binary_stat(image):
    hist = cv.calcHist([image], [0], None, [2], [0, 256])
    black, white = np.array(np.resize(hist, 2), dtype=int)
    return white, black


# clockwise
def rotate_bound(image, degree):
    h, w = image.shape
    theta = np.abs(degree) * np.pi/180
    width = int(w * np.cos(theta) + h * np.sin(theta))
    height = int(h * np.cos(theta) + w * np.sin(theta))
    canvas = np.zeros((height, width), dtype=np.uint8)
    upper = (height-h)//2
    left = (width-w)//2
    canvas[upper:upper+h, left:left+w] = image
    matrix = cv.getRotationMatrix2D((width//2, height//2), -degree, 1)
    canvas = cv.warpAffine(canvas, matrix, (width, height))
    return canvas


def hough_transform(image):
    correct_angle = np.pi / 180 * 46  # 只做45度以内的倾斜矫正
    min_angle = np.pi/2 - correct_angle
    max_angle = np.pi - min_angle

    dil_image = filter.add_dilate(image)  # 膨胀
    edges = cv.Canny(dil_image, 50, 150)  # 提取边缘
    # cv.imshow('edges', edges)
    # cv.waitKey()

    lines = cv.HoughLines(edges, 1, np.pi / 720, 100)
    all_valid_angles = []
    display = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines[:20]:  # 取可能性最大的二十条线
            rho, theta = line[0]
            if min_angle <= theta <= max_angle:
                all_valid_angles.append(theta)
            x0 = int(rho * np.cos(theta))
            y0 = int(rho * np.sin(theta))
            x1 = int(x0 + 2000*np.sin(theta))
            y1 = int(y0 - 2000*np.cos(theta))
            x2 = int(x0 - 2000 * np.sin(theta))
            y2 = int(y0 + 2000 * np.cos(theta))
            cv.line(display, (x2, y2), (x1, y1), (255, 0, 0))
    inclined_angle = (np.pi / 2 - np.average(all_valid_angles)) * 180 / np.pi if len(all_valid_angles) else 0
    inclined_angle = np.around(inclined_angle, 1)  # 精确到0.1度
    print('inclined angle:', inclined_angle)

    if inclined_angle != 0:
        image = rotate_bound(image, inclined_angle)

    return image, inclined_angle


# 垂直切割先水平腐蚀再竖直膨胀
def get_vertical_split(image):
    open_image = filter.add_erode(image, size=(2, 1))
    open_image = filter.add_dilate(open_image, size=(2, 4))
    # open_image = image
    # cv.imshow('horizon', open_image)
    # cv.waitKey()
    min_width = 3
    min_pixel = 1
    _, cols = open_image.shape
    stat = np.sum(open_image == 255, axis=0, dtype=int)

    height, width = image.shape
    col_range_list = []
    is_begin = False
    start = 0
    for i in range(cols):
        if not is_begin and stat[i] >= min_pixel:
            start = i
            is_begin = True
        elif is_begin and stat[i] < min_pixel:
            if i - start >= min_width:
                col_range = (start, i)  # 考虑到有腐蚀，左右多截一像素
                if (i-start)/height < 0.9 and len(col_range_list) > 0:
                    last_start, last_end = col_range_list[-1]
                    if (last_end-last_start)/height < 0.9 and (i-last_start)/height < 1.2 and start-last_end <= min_width:
                        col_range_list.pop()
                        col_range = (last_start, i)
                col_range_list.append(col_range)
                # cv.imshow('char', open_image[:, col_range[0]:col_range[1]])
                # cv.waitKey()
            is_begin = False
    return col_range_list


# 水平切割先竖直腐蚀再横向膨胀
def get_horizon_split(image):
    open_image = filter.add_erode(image, size=(1, 2))
    open_image = filter.add_dilate(open_image, size=(4, 2))
    # open_image = image
    min_height = 12
    min_pixel = 5
    rows, cols = open_image.shape
    print('statistic ...')
    stat = np.sum(open_image == 255, axis=1, dtype=int)

    # 插值后检测波谷效果不太好 暂时仅考虑简单纯文本
    # print('drawing ...')
    # x = [i for i in range(rows)]
    # plt.subplot(121)
    # plt.barh(x, [d for d in reversed(stat)], height=1)
    # cubic = spi.interp1d(x, stat, kind=3)
    # plt.subplot(122)
    # y = cubic(x)
    # plt.plot(y)
    # peaks, _ = find_peaks(-y, distance=min_height)
    # plt.plot(peaks, y[peaks], 'x')
    # plt.show()

    row_range_list = []
    is_begin = False
    start = 0
    for i in range(rows):
        if not is_begin and stat[i] >= min_pixel:
            start = i
            is_begin = True
        elif is_begin and (stat[i] < min_pixel or i == rows-1):  # 最后一行也有像素
            if i - start >= min_height:
                row_range_list.append((start, i))
            is_begin = False
    return row_range_list


def get_vertical_split_alpha(image):
    min_width = 1
    min_pixel = 1
    rows, cols = image.shape
    stat = np.sum(image == 255, axis=0, dtype=int)

    col_range_list = []
    is_begin = False
    start = 0
    for i in range(cols):
        if not is_begin and stat[i] >= min_pixel:
            start = i
            is_begin = True
        elif is_begin and stat[i] < min_pixel:
            if i - start >= min_width:
                col_range_list.append((start, i))
            is_begin = False

    return col_range_list


def char_resize(image, size=(64, 64)):
    ori_height, ori_width = image.shape
    height, width = size
    w_ratio = width/ori_width
    h_ratio = height/ori_height

    min_ratio = min(w_ratio, h_ratio)
    image = cv.resize(image, (int(ori_width*min_ratio), int(ori_height*min_ratio)))

    h, w = image.shape
    canvas = np.zeros(size, dtype=np.uint8)
    if w_ratio > h_ratio:
        diff = int((width-w) / 2)
        canvas[:h, diff:diff+w] = image
    else:
        diff = int((height-h) / 2)
        canvas[diff:diff+h, :w] = image
    # cv.imshow('canvas', canvas)
    # cv.waitKey()
    return canvas


def get_predict_image(path, size=(64, 64), alpha=False, noise=False):
        image = read_image(path)
        bin_image, origin = filter.binary_image_otsu(image)
        # cv.imshow('before filter', bin_image)
        if noise:
            bin_image = filter.mean_filter(bin_image)
        # cv.imshow('filter', bin_image)
        bin_image, inclined_angle = hough_transform(bin_image)   # 旋转后不是二值图
        bin_image, _ = filter.binary_image_otsu(bin_image)
        if inclined_angle != 0:
            origin = rotate_bound(origin, inclined_angle)

        # cv.imshow('binary', bin_image)
        # cv.waitKey()

        row_range_list = get_horizon_split(bin_image)

        valid_images = []
        image_box_list = []
        for upper, bottom in row_range_list:
            row_image = bin_image[upper:bottom]
            col_range_list = get_vertical_split(row_image) if not alpha else get_vertical_split_alpha(row_image)
            for left, right in col_range_list:
                image_box_list.append((left, upper, right, bottom))

        for left, upper, right, bottom in image_box_list:
            if not noise:
                valid_images.append(origin[upper:bottom, left:right])
            else:
                valid_images.append(bin_image[upper:bottom, left:right])

        regularized_image = []
        for image in valid_images:
            # cv.imshow('test', image)
            # cv.waitKey()
            character = char_resize(image, size=size)
            character = np.asarray(character, dtype=np.float32)/255
            regularized_image.append(character)
        return regularized_image


def main():
    batch = get_predict_image("input_image/test19.jpg", alpha=False)
    for image in batch:
        cv.imshow('char', np.array(image*255, dtype=np.uint8))
        cv.waitKey()
    exit()
    # save_combined_image('train_data/data_test/', image_list, char_size=(64, 64), group_size=(16, 16))


if __name__ == '__main__':
    main()
