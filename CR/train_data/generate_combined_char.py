from PIL import Image, ImageFont, ImageDraw
import cv2 as cv
import numpy as np
import time

import os
import sys
sys.path.append('../')
import get_integrate
import get_chinese


class ImageEnhance(object):
    def __init__(self, noise=True, dilate=True, erode=True):
        self.noise = noise
        self.dilate = dilate
        self.erode = erode

    @classmethod
    def add_noise(cls, img):
        noise_num = np.random.randint(3, 25)
        img = np.array(img)
        for i in range(noise_num):    # 椒盐噪声
            temp_x = np.random.randint(0, img.shape[0])
            temp_y = np.random.randint(0, img.shape[1])
            img[temp_x][temp_y] = 255
        return img

    @classmethod
    def add_erode(cls, img):
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        img = cv.erode(img, kernel)
        return img

    @classmethod
    def add_random_erode(cls, img):
        erode_num = np.random.randint(15, 25)
        img = np.array(img)
        for i in range(erode_num):
            temp_x = np.random.randint(0, img.shape[0])
            temp_y = np.random.randint(0, img.shape[1])
            img[temp_x:temp_x+3, temp_y:temp_y+3] = 0
        return img

    @classmethod
    def add_dilate(cls, img):
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        img = cv.dilate(img, kernel)
        return img

    @classmethod
    def add_gaussian(cls, img):
        bg = np.random.normal(20, 5, img.shape)
        ret_img = bg + img
        ret_img[ret_img < 0] = 0
        ret_img[ret_img > 255] = 255
        return np.asarray(ret_img, dtype=np.uint8)

    def enhance(self, img_list):
        print('Enhancing ...')
        ret_img_list = []
        for img in img_list:
            ret_img = img
            if np.random.random() < 0.3:
                pass
            elif self.dilate and np.random.random() < 0.5:
                ret_img = self.add_dilate(img)
            elif self.erode:
                ret_img = self.add_erode(img)
            if np.random.random() < 0.5:   # 局部腐蚀
                ret_img = self.add_random_erode(ret_img)
            if np.random.random() < 0.8:   # 高斯噪声
                ret_img = self.add_gaussian(ret_img)
            ret_img_list.append(self.add_noise(ret_img))
        print('Done')
        return ret_img_list


# def char_resize(char_image, max_width, max_height):
#     canvas = Image.new('L', (max_width, max_height), 0)
#     width, height = char_image.size
#     print(char_image.size)
#     height = int(max_width / width * height)
#     optical_char = char_image.resize((max_width, height), Image.BILINEAR)
#     if height > max_height:
#         width = int(max_height / height * width)
#         print(width, max_height)
#         optical_char = char_image.resize((width, max_height), Image.BILINEAR)
#
#     width, height = optical_char.size
#     canvas.paste(optical_char, (int((max_width - width) / 2), int((max_height - height) / 2)))
#     return canvas


def char_resize(char_image, max_width, max_height):
    canvas = Image.new('L', (max_width, max_height), 0)
    width, height = char_image.size

    if width < height:
        optical_char = char_image.resize((int(width*max_height/height), max_height), Image.BILINEAR)
    else:
        optical_char = char_image.resize((max_width, int(height * max_width / width)), Image.BILINEAR)

    width, height = optical_char.size
    canvas.paste(optical_char, (int((max_width - width) / 2), int((max_height - height) / 2)))
    return canvas


def font2image(font_style, gb_list, char_size=(64, 64), max_angle=30, angle_step=2):
    width, height = char_size
    font_size = int((width+height)/2)
    canvas_width = int(1.5*width)       # 大一点避免旋转后被动裁剪
    canvas_height = int(1.5*height)

    image = Image.new('L', (canvas_width, canvas_height), 0)  # L=gray RGB=rgb
    # 创建Font对象:
    font = ImageFont.truetype('font/' + font_style, font_size)
    # 创建Draw对象:
    draw = ImageDraw.Draw(image)

    # 输出文字:
    optical_char_list = []
    char_info_list = []
    num_per_angle, amount = get_integrate.get_distribution(max_angle, angle_step, scale=20)
    print('Total num of all angle per char:', amount)
    for char, code in gb_list:
        if code % 100 == 0:
            print('[+]Font2image: ' + str(code/len(gb_list)*100) + '% ...')
        image.paste(0, (0, 0, canvas_width, canvas_height))  # erase
        draw.text((int((canvas_width-width)/2), int((canvas_height-height)/2)), char, font=font, fill=255)
        for i, angle in enumerate(range(-max_angle, max_angle+1, angle_step)):
            optical_char = image.rotate(angle, Image.BILINEAR)   # linear interpolation
            box = optical_char.getbbox()
            optical_char = optical_char.crop(box)
            optical_char = char_resize(optical_char, width, height)
            # optical_char = optical_char.resize((width, height), Image.ANTIALIAS)  # 对‘一’resize 效果不理想
            for _ in range(num_per_angle[i]):
                optical_char_list.append(np.array(optical_char))
                char_info_list.append(char)
    return optical_char_list, char_info_list


def punctuation2image(font_style, gb_list, char_size=(64, 64)):
    width, height = char_size
    font_size = int((width+height)/2)

    image = Image.new('L', char_size, 0)  # L=gray RGB=rgb
    # 创建Font对象:
    font = ImageFont.truetype('font/' + font_style, font_size)
    # 创建Draw对象:
    draw = ImageDraw.Draw(image)
    # 输出文字:
    optical_punc_list = []
    punc_info_list = []
    print('[+]Punctuation2image ...')
    for char, code in gb_list:
        image.paste(0, (0, 0, width, height))  # erase
        font_w, font_h = font.getsize(char)
        max_v = max(font_w, font_h)

        font_image = Image.new('L', (font_w, font_h), 0)
        font_draw = ImageDraw.Draw(font_image)
        font_draw.text((0, 0), char, font=font, fill=255)
        if max_v > char_size[0]:
            font_image = font_image.resize((int(font_w * width / max_v), int(font_h * height / max_v)), Image.BILINEAR)
            font_w, font_h = font_image.size

        # 提取左右边界
        col_sum = np.sum(np.array(font_image, dtype=np.uint8) > 0, axis=0)
        left = 0
        right = font_w
        for i in range(1, font_w - 1):
            if col_sum[i] > 0 and col_sum[i - 1] == 0:
                left = i
            if col_sum[i] > 0 and col_sum[i + 1] == 0:
                right = i + 1
                break
        font_image = font_image.crop((left, 0, right, font_h))
        font_w, font_h = font_image.size

        left = int((width-font_w)/2)
        upper = int((height-font_h)/2)
        if font_image is None:
            draw.text((left, upper), char, font=font, fill=255)
        else:
            image.paste(font_image, (left, upper))

        optical_punc_list.append(np.array(image))
        punc_info_list.append(char)
    return optical_punc_list, punc_info_list


def image_paste(target, source, box):
    left, upper, right, lower = box
    target[upper:lower, left:right] = source


# 共72*2*13*3355张图片 分为N张256*256的图片集以及一张剩余的图片集 以减少IO消耗
def save_combined_image(path, all_char_list, char_size=(64, 64), group_size=(256, 256)):
    order = 1
    curr_num = 0
    label_group = []
    total_row, total_col = group_size
    width, height = char_size
    max_num_per_group = total_col*total_row
    total_group_num = np.ceil(len(all_char_list)/(total_row*total_col))
    print('Total group num:', total_group_num)
    container = np.zeros((total_row * height, total_col * width), dtype=np.uint8)
    for image, label in all_char_list:
        row = int(curr_num / total_col)
        col = int(curr_num % total_col)
        image_paste(container, image, (col*width, row*height, (col+1)*width, (row+1)*height))
        label_group.append(label)
        curr_num += 1
        if curr_num == max_num_per_group:
            curr_num = 0
            save_image_and_label(path, container, label_group, order, group_size)
            label_group = []
            container = np.zeros((total_row * height, total_col * width), dtype=np.uint8)
            order += 1
            print('Saving: {:.2f}% ...'.format(order/total_group_num*100))
    if curr_num < max_num_per_group:   # 最后一组
        save_image_and_label(path, container, label_group, order, group_size)
    print('Save Done')


def save_image_and_label(path, image, label, order, group_size=(128, 128)):
    total_row, total_col = group_size
    cv.imencode('.jpg', image)[1].tofile(path + 'images_{}.jpg'.format(order))
    with open(path + 'labels_{}.txt'.format(order), 'wt', encoding='utf8') as file:
        for i in range(total_row):
            for j in range(total_col):
                if i*total_col + j >= len(label):
                    file.write('\n')
                    return
                file.write(label[i*total_col + j])
            file.write('\n')


def main():
    gb_list = get_chinese.get_primary_gb()
    print('Total char:', len(gb_list))
    gb_char_list = gb_list[:-42]
    gb_punc_list = gb_list[-42:]

    font_style_list = ['simhei.ttf', 'simkai.ttf', 'simsun.ttc', 'msyh.ttc', 'STXINWEI.TTF', 'SIMLI.TTF', 'FZSTK.TTF',
                       'Deng.ttf', 'STXINGKA.TTF', 'FZYTK.TTF', 'simfang.ttf', 'SIMYOU.TTF', 'STSONG.TTF']

    all_char_list = []
    for i, style in enumerate(font_style_list):
        print(style, '{}/{}'.format(i+1, len(font_style_list)))
        optical_char_list, char_info_list = font2image(style, gb_char_list, char_size=(64, 64), angle_step=3)
        optical_char_list.extend(ImageEnhance().enhance(optical_char_list))
        char_info_list *= 2
        all_char_list.extend(zip(optical_char_list, char_info_list))

        optical_punc_list, punc_info_list = punctuation2image(style, gb_punc_list, char_size=(64, 64))
        optical_punc_list *= 5          # 由于没有旋转，扩大5倍
        punc_info_list *= 5
        enhance1 = ImageEnhance(erode=False).enhance(optical_punc_list)
        enhance2 = ImageEnhance(erode=False).enhance(optical_punc_list)
        optical_punc_list.extend(enhance1 + enhance2)
        punc_info_list *= 3
        all_char_list.extend(zip(optical_punc_list, punc_info_list))

    print('Shuffle ...')
    np.random.shuffle(all_char_list)
    test_data_ratio = 0.02
    test_data_length = int(len(all_char_list)*test_data_ratio)

    print('Total Length:', len(all_char_list))
    print('Train data length:', len(all_char_list)-test_data_length)
    print('Test data length:', test_data_length)
    path_train = 'data_train/'
    path_test = 'data_test/'
    save_combined_image(path_train, all_char_list[:-test_data_length], char_size=(64, 64), group_size=(256, 256))
    save_combined_image(path_test, all_char_list[-test_data_length:], char_size=(64, 64), group_size=(256, 256))
    print('Total char:', len(gb_list))


if __name__ == '__main__':
    begin = time.time()
    main()
    print('Total Time:', (time.time() - begin) / 60)
