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


def get_primer_gb():
    start = 0xB0A1
    end = 0xD7FA
    gb_list = []
    gb_id = 0
    for i in range(start, end):
        if (i & 0xF0) >> 4 < 0xA or i & 0xF == 0x0 or i & 0xF == 0xF:
            continue
        character = str(i.to_bytes(length=2, byteorder='big'), 'gb2312')
        gb_list.append((character, gb_id))
        gb_id += 1
    return gb_list


def get_batch(size, char_size=(64, 64), group_size=(256, 256), total_num=3355*13*72*2, one_hot_length=None):
    gb_list = get_primer_gb()
    label_map = {}
    for char, gb_id in gb_list:
        label_map[char] = gb_id
    print(label_map)

    curr_group_remain = 0
    curr_images = []
    curr_labels = []
    curr_pointer = 0
    next_order = 1

    loaded_num = 0

    while loaded_num < total_num:
        if curr_group_remain == 0:
            curr_images, curr_labels = load_image_group(next_order, char_size=char_size, group_size=group_size)
            curr_pointer = 0
            curr_group_remain = len(curr_labels)
            next_order += 1

        if size > curr_group_remain:
            ret_image_list = curr_images[curr_pointer:]
            ret_label_list = curr_labels[curr_pointer:]
            if loaded_num + size < total_num:
                curr_images, curr_labels = load_image_group(next_order, char_size=char_size, group_size=group_size)
                curr_pointer = 0
                curr_group_remain = len(curr_labels)
                next_order += 1

                diff = size - len(ret_image_list)
                ret_image_list.extend(curr_images[:diff])
                ret_label_list.extend(curr_labels[:diff])

                curr_pointer += diff
                curr_group_remain -= diff
                loaded_num += size
            else:
                loaded_num = total_num
        else:
            ret_image_list = curr_images[curr_pointer: curr_pointer + size]
            ret_label_list = curr_labels[curr_pointer: curr_pointer + size]

            curr_pointer += size
            curr_group_remain -= size
            loaded_num += size

        ret_label_list = [label_map[label] for label in ret_label_list]

        if one_hot_length is not None:
            yield np.array(ret_image_list, dtype=np.float32)/255, get_one_hot(ret_label_list, one_hot_length)
        else:
            yield np.array(ret_image_list, dtype=np.float32)/255, np.array(ret_label_list)


def load_image_group(order, group_size=(256, 256), char_size=(64, 64)):
    print('Loading group {} ...'.format(order))
    _, total_col = group_size
    width, height = char_size
    directory = 'train_data/train_data_batch/'
    directory = 'train_data/train_data_new/'
    directory = 'train_data/train_data_test/'
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


def get_one_hot(arr, dim):
    matrix = np.zeros((len(arr), dim), dtype=int)
    for i, col in enumerate(arr):
        matrix[i][col] = 1
    return matrix


def main():
    gb_list = get_primer_gb()
    label_id = {}
    id_label = {}
    for char, gb_id in gb_list:
        label_id[char] = gb_id
        id_label[gb_id] = char

    batch_size = 50
    batch_gen = get_batch(batch_size, total_num=3355*13*72*2, group_size=(256, 256))
    try:
        while True:
            batch = next(batch_gen)
            images, labels = batch
            print('batch size:', len(images), len(labels))
            for i in range(len(images)):
                pos = np.argmax(labels[i])
                print(pos, id_label[pos], i)
                cv.imshow('image', images[i])
                cv.waitKey()
    except StopIteration:
        pass


if __name__ == '__main__':
    main()
