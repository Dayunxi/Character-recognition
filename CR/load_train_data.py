import cv2 as cv
import numpy as np

import get_chinese


# length should less than multiply(group_size)
def get_batch(path, length, total_num, char_size=(64, 64), group_size=(128, 128), one_hot_length=None):
    gb_list = get_chinese.get_primary_gb()
    label_map = {}
    for char, gb_id in gb_list:
        label_map[char] = gb_id
    # print(label_map)

    curr_group_remain = 0
    curr_images = []
    curr_labels = []
    curr_pointer = 0
    next_order = 1

    loaded_num = 0

    while loaded_num < total_num:
        if curr_group_remain == 0:
            del curr_images, curr_labels
            curr_images, curr_labels = load_image_group(path, next_order, char_size=char_size, group_size=group_size)
            curr_pointer = 0
            curr_group_remain = len(curr_labels)
            next_order += 1

        if length > curr_group_remain:
            ret_image_list = curr_images[curr_pointer:]
            ret_label_list = curr_labels[curr_pointer:]
            if loaded_num + length < total_num:
                del curr_images, curr_labels
                curr_images, curr_labels = load_image_group(path, next_order, char_size=char_size, group_size=group_size)
                curr_pointer = 0
                curr_group_remain = len(curr_labels)
                next_order += 1

                diff = length - len(ret_image_list)
                ret_image_list.extend(curr_images[:diff])
                ret_label_list.extend(curr_labels[:diff])

                curr_pointer += diff
                curr_group_remain -= diff
                loaded_num += length
            else:
                loaded_num = total_num
        else:
            ret_image_list = curr_images[curr_pointer: curr_pointer + length]
            ret_label_list = curr_labels[curr_pointer: curr_pointer + length]

            curr_pointer += length
            curr_group_remain -= length
            loaded_num += length

        ret_label_list = [label_map[label] for label in ret_label_list]

        if one_hot_length is not None:
            yield np.array(ret_image_list, dtype=np.float32)/255, get_one_hot(ret_label_list, one_hot_length)
        else:
            yield np.array(ret_image_list, dtype=np.float32)/255, np.array(ret_label_list)


def load_image_group(path, order, char_size=(64, 64), group_size=(128, 128)):
    print('Loading group {} ...'.format(order))
    _, total_col = group_size
    width, height = char_size

    grouped_image = np.array(cv.imread(path + 'images_{}.jpg'.format(order), cv.IMREAD_GRAYSCALE))
    images = []
    labels = []
    with open(path + 'labels_{}.txt'.format(order), 'rt', encoding='utf8') as file:
        lines = file.readlines()
        for line in lines:
            labels.extend(list(line.strip()))
    for i in range(len(labels)):
        row = int(i / total_col)
        col = i % total_col
        image = grouped_image[row*height:(row+1)*height, col*width:(col+1)*width]
        images.append(image)
    print(len(labels), len(images))
    print('Load Done')
    return images, labels


def get_one_hot(arr, dim):
    matrix = np.zeros((len(arr), dim), dtype=int)
    for i, col in enumerate(arr):
        matrix[i][col] = 1
    return matrix


# 二值化
def binary_image(image):
    # 原图黑字白底
    return image
    # threshold, ret = cv.threshold(image, 0, 255, cv.THRESH_OTSU)
    # return ret


def main():
    label_id, id_label = get_chinese.get_char_map()

    batch_size = 50
    batch_gen = get_batch('train_data/data_test/', batch_size, total_num=3755*13*72*2, group_size=(256, 256))
    try:
        while True:
            batch = next(batch_gen)
            images, labels = batch
            print('batch size:', len(images), len(labels))
            for i in range(len(images)):
                pos = np.argmax(labels[i])
                print(pos, id_label[pos], i)
                cv.imshow('image', np.array(images[i], dtype=np.uint8))
                cv.waitKey()
    except StopIteration:
        pass


if __name__ == '__main__':
    main()
