import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as spi
from scipy.signal import find_peaks

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


def binary_stat(image):
    hist = cv.calcHist([image], [0], None, [2], [0, 256])
    black, white = np.array(np.resize(hist, 2), dtype=int)
    return white, black


# 二值化，并转为黑底白字
def binary_image(image):
    # 原图黑字白底
    ret = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 31, 10)
    white_cnt, black_cnt = binary_stat(ret)
    origin = image
    if white_cnt > black_cnt:
        print(white_cnt, black_cnt)
        ret = cv.bitwise_not(ret)
        origin = cv.bitwise_not(image)
    return ret, origin


def binary_image_otsu(image):
    threshold, ret = cv.threshold(image, 0, 255, cv.THRESH_OTSU)
    return ret


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


def add_erode(img, size=(2, 2)):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, size)
    img = cv.erode(img, kernel)
    return img


def add_dilate(img, size=(5, 5)):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, size)
    img = cv.dilate(img, kernel)
    return img


def hough_transform(image):
    correct_angle = np.pi / 180 * 46  # 只做45度以内的倾斜矫正
    min_angle = np.pi/2 - correct_angle
    max_angle = np.pi - min_angle

    dil_image = add_dilate(image)  # 膨胀
    edges = cv.Canny(dil_image, 50, 150)  # 提取边缘
    # cv.imshow('edges', edges)

    lines = cv.HoughLines(edges, 1, np.pi / 720, 100)
    all_valid_angles = []
    # display = np.array(image)
    display = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    if lines is not None:
        # print(image.shape)
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
    print('inclined angle', inclined_angle)
    # cv.imshow('display', display)

    if inclined_angle != 0:
        image = rotate_bound(image, inclined_angle)

    return image, inclined_angle


# 垂直切割先水平腐蚀再竖直膨胀
def get_vertical_split(image, origin):
    open_image = add_erode(image, size=(2, 1))
    # cv.imshow('erode', open_image)
    # cv.waitKey()
    open_image = add_dilate(open_image, size=(2, 4))

    min_width = 3
    min_pixel = 1
    rows, cols = open_image.shape
    stat = np.sum(open_image == 255, axis=0, dtype=int)

    height, width = image.shape
    crop_info_list = []
    is_begin = False
    start = 0
    for i in range(cols):
        if not is_begin and stat[i] >= min_pixel:
            start = i
            is_begin = True
        elif is_begin and stat[i] < min_pixel:
            if i - start >= min_width:
                cropped_info = (origin[:, (start-1 if start else 0):i+1], start, i)  # 考虑到有腐蚀，左右多截一像素

                if (i-start)/height < 0.9 and len(crop_info_list) > 0:
                    _, last_start, last_end = crop_info_list[-1]
                    if (last_end-last_start)/height < 0.9 and (i-last_start)/height < 1.2:
                        crop_info_list.pop()
                        # 考虑到有腐蚀，左右多截一像素
                        cropped_info = (origin[:, (last_start-1 if last_start else 0):i+1], last_start, i)
                crop_info_list.append(cropped_info)
            is_begin = False
    col_split = [image[0] for image in crop_info_list]
    return col_split


# 水平切割先竖直腐蚀再横向膨胀
def get_horizon_split(image, origin):
    open_image = add_erode(image, size=(1, 2))
    open_image = add_dilate(open_image, size=(4, 2))

    min_height = 12
    min_pixel = 1
    rows, cols = open_image.shape
    print('statistic ...')
    stat = np.sum(open_image == 255, axis=1, dtype=int)

    # labels = [num for num in range(rows, 1, -1)]
    # plt.subplot(121)
    # plt.imshow(np.array(image, dtype=np.uint8))
    # plt.subplot(122)
    # plt.barh(range(500), [d for d in reversed(stat[100:600])], height=1)
    # plt.yticks([])
    # plt.show()
    # exit()

    # 效果不太好 暂时仅考虑简单纯文本
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

    row_split = []
    origin_split = []
    is_begin = False
    for i in range(rows):
        if not is_begin and stat[i] >= min_pixel:
            start = i
            is_begin = True
        elif is_begin and stat[i] < min_pixel:
            if i - start >= min_height:
                row_split.append(image[start:i, :])
                origin_split.append(origin[start:i, :])
            is_begin = False
    return row_split, origin_split


def image_paste(target, source, box):
    left, upper, right, lower = box
    target[upper:lower, left:right] = source


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


def get_predict_image(path, size=(64, 64)):
        image = read_image(path)
        image, origin = binary_image(image)
        image, inclined_angle = hough_transform(image)
        if inclined_angle != 0:
            origin = rotate_bound(origin, inclined_angle)

        horizon_images, origin_horizon = get_horizon_split(image, origin)
        # print(len(horizon_images))
        valid_images = []
        for row_image, origin in zip(horizon_images, origin_horizon):
            images = get_vertical_split(row_image, origin)
            valid_images.extend(images)
        regularized_image = []
        for image in valid_images:
            # cv.imshow('test', image)
            # cv.waitKey()
            character = char_resize(image, size=size)
            character = np.array((character), dtype=np.float32)/255
            regularized_image.append(character)
        return regularized_image


def main():
    batch = get_predict_image("input_image/test16.jpg")
    exit()
    # save_combined_image('train_data/data_test/', image_list, char_size=(64, 64), group_size=(16, 16))


if __name__ == '__main__':
    main()
