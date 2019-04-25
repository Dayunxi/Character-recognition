import cv2 as cv
import numpy as np


def add_erode(img, size=(2, 2)):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, size)
    img = cv.erode(img, kernel)
    return img


def add_dilate(img, size=(5, 5)):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, size)
    img = cv.dilate(img, kernel)
    return img


def binary_stat(image):
    hist = cv.calcHist([image], [0], None, [2], [0, 256])
    black, white = np.array(np.resize(hist, 2), dtype=int)
    return white, black


# 二值化，并转为黑底白字
def binary_image_ada(image):
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
    white_cnt, black_cnt = binary_stat(ret)
    origin = image
    if white_cnt > black_cnt:
        print(white_cnt, black_cnt)
        ret = cv.bitwise_not(ret)
        origin = cv.bitwise_not(image)
    return ret, origin


def mean_filter(img):
    img = cv.medianBlur(img, 3)
    return img


def image_filter(image):
    cv.imshow('origin', image)
    # image = gaussian_filter(image)
    # cv.imshow('filter', image)
    image = binary_image_otsu(image)
    cv.imshow('binary', image)
    image = mean_filter(image)
    cv.imshow('filter', image)
    open_image = add_erode(image, size=(1, 2))
    open_image = add_dilate(open_image, size=(4, 2))
    cv.imshow('open', open_image)
    cv.waitKey()


def main():
    image = cv.imread('input_image/test24.jpg', cv.IMREAD_GRAYSCALE)
    # image = resize(image)
    print(image.shape)
    image_filter(image)


if __name__ == '__main__':
    main()
