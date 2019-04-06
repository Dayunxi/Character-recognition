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
    white = 0
    black = 0
    for row in image:
        for pixel in row:
            if pixel == 255:
                white += 1
            elif pixel == 0:
                black += 1
    return white, black


# 二值化，并转为黑底白字
def binary_image(image):
    # 原图黑字白底
    ret = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 31, 10)
    white_cnt, black_cnt = binary_stat(ret)
    if white_cnt > black_cnt:
        print(white_cnt, black_cnt)
        ret = cv.bitwise_not(ret)
    return ret


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


def add_noise(img, min_num=20, max_num=25):
    noise_num = np.random.randint(min_num, max_num)
    img = np.array(img)
    for i in range(noise_num):    # 椒盐噪声
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img


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

    lines = cv.HoughLines(edges, 1, np.pi / 720, 100)
    all_valid_angles = []
    display = np.array(image)
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
            cv.line(display, (x2, y2), (x1, y1), 255)
    inclined_angle = (np.pi / 2 - np.average(all_valid_angles)) * 180 / np.pi if len(all_valid_angles) else 0
    inclined_angle = np.around(inclined_angle, 1)  # 精确到0.1度
    print('inclined angle', inclined_angle)
    if inclined_angle != 0:
        image = rotate_bound(image, inclined_angle)
    return image


# 垂直切割先水平腐蚀再竖直膨胀
def get_vertical_split(image):
    open_image = add_erode(image, size=(2, 1))
    # cv.imshow('erode', open_image)
    # cv.waitKey()
    open_image = add_dilate(open_image, size=(2, 4))

    min_width = 3
    min_pixel = 1
    rows, cols = open_image.shape
    stat = np.zeros(cols, dtype=int)
    for i in range(rows):
        for j in range(cols):
            if open_image[i][j] == 255:
                stat[j] += 1

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
                cropped_info = (image[:, (start-1 if start else 0):i+1], start, i)  # 考虑到有腐蚀，左右多截一像素
                if (i-start)/height < 0.9 and len(crop_info_list) > 1:
                    _, last_start, last_end = crop_info_list[-1]
                    if (last_end-last_start)/height < 0.9 < (i-last_start)/height < 1.2:
                        del crop_info_list[-1]
                        # 考虑到有腐蚀，左右多截一像素
                        cropped_info = (image[:, (last_start-1 if last_start else 0):i+1], last_start, i)
                        # cv.imshow('conjunction', cropped_info[0])
                        # cv.waitKey()
                crop_info_list.append(cropped_info)
            is_begin = False
    col_split = [image[0] for image in crop_info_list]
    return col_split


# 水平切割先竖直腐蚀再横向膨胀
def get_horizon_split(image):
    # image = add_noise(image, 1500, 2000)
    # cv.imshow('origin', image)
    # cv.waitKey()
    open_image = add_erode(image, size=(1, 2))
    # cv.imshow('erode', image)
    # cv.waitKey()
    open_image = add_dilate(open_image, size=(4, 2))
    # cv.imshow('dilate', image)
    # cv.waitKey()

    min_height = 12
    min_pixel = 1
    rows, cols = open_image.shape
    stat = np.zeros(rows, dtype=int)
    print('statistic ...')
    for i in range(rows):
        for j in range(cols):
            if open_image[i][j] == 255:
                stat[i] += 1

    # plt.barh(range(rows), [d for d in reversed(stat)], height=1)
    # plt.show()

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
    is_begin = False
    for i in range(rows):
        if not is_begin and stat[i] >= min_pixel:
            start = i
            is_begin = True
        elif is_begin and stat[i] < min_pixel:
            if i - start >= min_height:
                row_split.append(image[start:i, :])
            is_begin = False
    return row_split


def image_paste(target, source, box):
    left, upper, right, lower = box
    target[upper:lower, left:right] = source


def save_combined_image(path, all_char_list, char_size=(64, 64), group_size=(16, 16)):
    order = 1
    curr_num = 0
    total_row, total_col = group_size
    width, height = char_size
    max_num_per_group = total_col*total_row
    total_group_num = np.ceil(len(all_char_list)/(total_row*total_col))
    print('Total group num:', total_group_num)
    container = np.zeros((total_row * height, total_col * width), dtype=np.uint8)
    for image in all_char_list:
        row = int(curr_num / total_col)
        col = int(curr_num % total_col)
        image_paste(container, image, (col*width, row*height, (col+1)*width, (row+1)*height))
        curr_num += 1
        if curr_num == max_num_per_group:
            curr_num = 0
            cv.imencode('.jpg', container)[1].tofile(path + 'images_{}.jpg'.format(order))
            container = np.zeros((total_row * height, total_col * width), dtype=np.uint8)
            order += 1
            print('Saving: {:.2f}% ...'.format(order/total_group_num*100))
    if curr_num < max_num_per_group:   # 最后一组
        cv.imencode('.jpg', container)[1].tofile(path + 'images_{}.jpg'.format(order))
    print('Save Done')


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
        image = binary_image(image)
        # image = hough_transform(image)
        # cv.imshow('origin', image)
        # cv.waitKey()
        # image = add_erode(image, size=(2, 1))
        # cv.imshow('erode', image)
        # cv.waitKey()
        # image = add_dilate(image, size=(2, 4))
        # cv.imshow('dilate', image)
        # cv.waitKey()

        horizon_images = get_horizon_split(image)
        # print(len(horizon_images))
        valid_images = []
        for item in horizon_images:
            images = get_vertical_split(item)
            valid_images.extend(images)
        regularized_image = []
        for image in valid_images:
            character = char_resize(image, size=size)
            character = np.array(binary_image_otsu(character), dtype=np.float32)/255
            regularized_image.append(character)
        return regularized_image


def main():
    exit()
    image = read_image("input_image/test0.jpg")
    image = binary_image(image)
    image = hough_transform(image)
    cv.imwrite('input_image/result.jpg', image)
    # image = rotate_bound(image, 30)
    # cv.imshow('binary', image)
    # cv.waitKey()

    horizon_images = get_horizon_split(image)
    print(len(horizon_images))
    valid_images = []
    for item in horizon_images:
        images = get_vertical_split(item)
        valid_images.extend(images)
    # print(valid_images)
    image_list = []
    for image in valid_images:
        height, width = image.shape
        print(image.shape, width/height)
        # image = cv.resize(image, (64, 64))
        # image_list.append(image)
        cv.imshow('show', image)
        cv.waitKey()
    # save_combined_image('train_data/data_test/', image_list, char_size=(64, 64), group_size=(16, 16))


if __name__ == '__main__':
    main()
