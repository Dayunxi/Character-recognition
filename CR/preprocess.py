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
    # cv.imshow('test', canvas)
    # cv.waitKey()
    canvas = cv.warpAffine(canvas, matrix, (width, height))
    return canvas


def add_erode(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    img = cv.erode(img, kernel)
    return img


def add_dilate(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    img = cv.dilate(img, kernel)
    return img


def hough_transform(image):
    correct_angle = np.pi / 180 * 46  # 只做45度以内的倾斜矫正
    min_angle = np.pi/2 - correct_angle
    max_angle = np.pi - min_angle
    dil_image = add_dilate(image)  # 膨胀
    # cv.imshow('dil', dil_image)
    # cv.waitKey()
    edges = cv.Canny(dil_image, 50, 150)  # 提取边缘
    # cv.imshow('edges', edges)
    # cv.waitKey()
    lines = cv.HoughLines(edges, 1, np.pi / 720, 100)
    all_valid_angles = []
    display = np.array(image)
    if lines is not None:
        print(image.shape)
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
    # cv.imshow('hough', display)
    # cv.waitKey()
    if inclined_angle != 0:
        image = rotate_bound(image, inclined_angle)
    return image


# 获取图片有效区间
def get_vertical_stat(image):
    min_width = 8
    min_pixel = 1
    rows, cols = image.shape
    stat = np.zeros(cols, dtype=int)
    for i in range(rows):
        for j in range(cols):
            if image[i][j] == 255:
                stat[j] += 1
    col_split = []
    is_begin = False
    for i in range(cols):
        if not is_begin and stat[i] >= min_pixel:
            start = i
            is_begin = True
        elif is_begin and stat[i] < min_pixel:
            if i - start >= min_width:
                col_split.append(image[:, start:i])
            is_begin = False
    return col_split


def get_horizon_stat(image):
    min_height = 12
    min_pixel = 1
    rows, cols = image.shape
    stat = np.zeros(rows, dtype=int)
    for i in range(rows):
        for j in range(cols):
            if image[i][j] == 255:
                stat[i] += 1

    plt.barh(range(rows), [d for d in reversed(stat)], height=1)
    plt.show()

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


# def get_valid_range(image, mode):
#     min_range = 2
#     rows, cols = image.shape
#     length = 0
#     stat = []
#     if mode == 'horizon':
#         length = rows
#         stat = np.zeros(length, dtype=int)
#         for i in range(rows):
#             for j in range(cols):
#                 if image[i][j] == 255:
#                     stat[i] += 1
#     elif mode == 'vertical':
#         length = cols
#         stat = np.zeros(length, dtype=int)
#         for i in range(rows):
#             for j in range(cols):
#                 if image[i][j] == 255:
#                     stat[j] += 1
#
#     print(mode, stat)
#
#     valid_range = []
#     threshold = 0
#     can_begin = True
#     start = 0
#     for i in range(length):
#         if can_begin and stat[i] > threshold:
#             start = i
#             can_begin = False
#         if not can_begin and stat[i] <= threshold:
#             if i - start >= min_range:
#                 valid_range.append((start, i))  # [start, end)
#             can_begin = True
#     print(valid_range)
#     valid_image = []
#     if mode == 'horizon':
#         for item in valid_range:
#             valid_image.append(image[item[0]:item[1]])
#     elif mode == 'vertical':
#         for item in valid_range:
#             valid_image.append(image[:, item[0]:item[1]])
#     return valid_image


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


def main():
    image = read_image("input_image/test2.jpg")
    image = binary_image(image)
    image = hough_transform(image)
    cv.imwrite('input_image/result.jpg', image)
    # image = rotate_bound(image, 30)
    cv.imshow('binary', image)
    cv.waitKey()

    horizon_images = get_horizon_stat(image)
    valid_images = []
    for item in horizon_images:
        images = get_vertical_stat(item)
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
