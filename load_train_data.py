from PIL import Image, ImageFont, ImageDraw
import os
import cv2 as cv
import numpy as np


def load_all_data():
    dir_list = os.listdir('train_data/character/')
    train_data = []
    for i, char in enumerate(dir_list):
        if i % 20 == 0:
            print('[+]Loading: ' + str(i/len(dir_list)*100) + '% ...')
        folder = 'train_data/character/' + char + '/'
        file_list = os.listdir(folder)
        # print(file_list)
        single_char_list = []
        for filename in file_list:
            path = folder + filename
            image = Image.open(path)
            single_char_list.extend(horizon_crop(image, 49))
        train_data.extend([(np.array(img), char) for img in single_char_list])
    print('Shuffling ...')
    np.random.shuffle(train_data)
    print('Done')
    return train_data


def horizon_crop(combined_image, num):
    image_list = []
    width, height = combined_image.size
    # print(width, height)
    w = int(width / num)
    for i in range(num):
        image_list.append(combined_image.crop((w*i, 0, w*(i+1), height)))
    return image_list


def get_batch(size):
    curr_group_remain = 0
    curr_images = []
    curr_labels = []
    curr_pointer = 0
    curr_order = 0

    loaded_num = 0
    total_num = 3355 * 7 * 49

    while loaded_num < total_num:
        if curr_group_remain == 0:
            curr_images, curr_labels = load_image_group(curr_order+1)
            curr_pointer = 0
            curr_group_remain = len(curr_labels)
            curr_order += 1

        if size > curr_group_remain:
            ret_image_list = curr_images[curr_pointer:]
            ret_label_list = curr_labels[curr_pointer:]
            if loaded_num + size < total_num:
                curr_images, curr_labels = load_image_group(curr_order+1)
                curr_pointer = 0
                curr_group_remain = len(curr_labels)
                curr_order += 1

                diff = size - len(ret_image_list)
                ret_image_list.extend(curr_images[:diff])
                ret_label_list.extend(curr_labels[:diff])

                curr_pointer += diff
                curr_group_remain -= diff
                loaded_num += size
        else:
            ret_image_list = curr_images[curr_pointer: curr_pointer + size]
            ret_label_list = curr_labels[curr_pointer: curr_pointer + size]

            curr_pointer += size
            curr_group_remain -= size
            loaded_num += size

        yield ret_image_list, ret_label_list


def load_image_group(order, group_size=(256, 256)):
    print('Loading group {} ...'.format(order))
    _, total_col = group_size
    width = 32
    height = 32
    directory = 'train_data/train_data_batch/'
    grouped_image = Image.open(directory + 'train_image_{}.jpg'.format(order))
    images = []
    labels = []
    with open(directory + 'train_label_{}.txt'.format(order), 'rt', encoding='utf8') as file:
        lines = file.readlines()
        for line in lines:
            labels.extend(list(line.strip()))
    for i in range(len(labels)):
        row = int(i / total_col)
        col = i % total_col
        image = grouped_image.crop((col*width, row*height, (col+1)*width, (row+1)*height))
        images.append(np.array(image))
    print(len(labels), len(images))
    print('Load Done')
    return images, labels


def main():
    batch = get_batch(int(256*256*2/5))
    for i in range(18):
        print(i)
        x, y = next(batch)
        print('batch size:', len(x), len(y))
        show_list = [0, 1, 2, -3, -2, -1]
        for j in show_list:
            print(y[j], j)
            cv.imshow('image', x[j])
            cv.waitKey()


if __name__ == '__main__':
    main()
