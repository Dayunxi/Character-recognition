from PIL import Image, ImageFont, ImageDraw
import cv2 as cv
import numpy as np
import os
import time


class ImageEnhance(object):
    def __init__(self, noise=True, dilate=True, erode=True):
        self.noise = noise
        self.dilate = dilate
        self.erode = erode

    @classmethod
    def add_noise(cls, img):
        noise_num = np.random.randint(1, 25)
        for i in range(noise_num):     # 添加点噪声
            temp_x = np.random.randint(0, img.shape[0])
            temp_y = np.random.randint(0, img.shape[1])
            img[temp_x][temp_y] = 255
        return img

    @classmethod
    def add_erode(cls, img):
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
        img = cv.erode(img, kernel)
        return img

    @classmethod
    def add_dilate(cls, img):
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
        img = cv.dilate(img, kernel)
        return img

    def enhance(self, img_list):
        print('Enhancing ...')
        for i in range(len(img_list)):
            im = img_list[i]
            if np.random.random() < 0.4:
                pass
            elif self.dilate and np.random.random() < 0.5:
                img_list[i] = self.add_dilate(im)
            elif self.erode:
                img_list[i] = self.add_erode(im)
            if self.noise and np.random.random() < 0.8:
                img_list[i] = self.add_noise(im)
        print('Done')


def get_primer_gb():
    start = 0xB0A1
    end = 0xD7FA
    gb_list = []
    gb_id = 1
    for i in range(start, end):
        if (i & 0xF0) >> 4 < 0xA or i & 0xF == 0x0 or i & 0xF == 0xF:
            continue
        character = str(i.to_bytes(length=2, byteorder='big'), 'gb2312')
        gb_list.append((character, gb_id))
        gb_id += 1
    return gb_list


def char_resize(char_image, data_width, data_height):
    canvas = Image.new('L', (data_width, data_height), 0)
    width, height = char_image.size
    height = int(32 / width * height)
    optical_char = char_image.resize((data_width, height), Image.BILINEAR)
    if height > 32:
        width = int(32 / height * width)
        optical_char = char_image.resize((width, data_height), Image.BILINEAR)

    width, height = optical_char.size
    canvas.paste(optical_char, (int((data_width - width) / 2), int((data_height - height) / 2)))
    # cv.imshow('resize', np.array(canvas))
    # cv.waitKey()
    return canvas


def font2image(font_style, gb_list):
    font_size = 32
    width = 32
    height = 32
    canvas_width = 50       # 大一点方便裁剪
    canvas_height = 50
    max_angle = 30
    angle_step = 2
    image = Image.new('L', (canvas_width, canvas_height), 0)  # L=gray RGB=rgb
    # 创建Font对象:
    font = ImageFont.truetype(font_style, font_size)
    # 创建Draw对象:
    draw = ImageDraw.Draw(image)

    # 输出文字:
    optical_char_list = []
    char_info_list = []
    # copy from get_integrate.py
    amount = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for char, code in gb_list:
        if code % 100 == 0:
            print('[+]Font2image: ' + str(code/len(gb_list)*100) + '% ...')
        image.paste(0, (0, 0, canvas_width, canvas_height))  # erase
        draw.text((5, 5), char, font=font, fill=255)
        for i, angle in enumerate(range(-max_angle, max_angle+1, angle_step)):
            optical_char = image.rotate(angle, Image.BILINEAR)   # linear interpolation
            box = optical_char.getbbox()
            optical_char = optical_char.crop(box)
            optical_char = char_resize(optical_char, width, height)
            # optical_char = optical_char.resize((width, height), Image.ANTIALIAS)  # 对‘一’resize 效果不理想
            for _ in range(amount[i]):
                optical_char_list.append(np.array(optical_char))
                char_info_list.append(char)
    return optical_char_list, char_info_list


def image_paste(target, source, box):
    left, upper, right, lower = box
    target[upper:lower, left:right] = source


# 共49*7*3355=1150765张图片 分为70张128*128的图片集以及一张剩余的图片集
def save_combined_image(all_char_list, group_size=(128, 128)):
    print('shuffle ...')
    np.random.shuffle(all_char_list)
    order = 1
    curr_num = 0
    label_group = []
    total_row, total_col = group_size
    max_num_per_group = total_col*total_row
    total_group_num = np.ceil(len(all_char_list)/(total_row*total_col))
    print('Total group num:', total_group_num)
    container = np.zeros((total_row * 32, total_col * 32), dtype=np.uint8)
    for image, label in all_char_list:
        row = int(curr_num / total_col)
        col = int(curr_num % total_col)
        image_paste(container, image, (col*32, row*32, col*32+32, row*32+32))
        label_group.append(label)
        curr_num += 1
        if curr_num == max_num_per_group:
            curr_num = 0
            save_image_and_label(container, label_group, order, group_size)
            label_group = []
            container = np.zeros((total_row * 32, total_col * 32), dtype=np.uint8)
            order += 1
            print('Saving: {:.2f}% ...'.format(order/total_group_num*100))
    if curr_num < max_num_per_group:
        save_image_and_label(container, label_group, order, group_size)
    print('Save Done')


def save_image_and_label(image, label, order, label_size=(128, 128)):
    total_row, total_col = label_size
    directory = 'train_data_batch/'
    cv.imencode('.jpg', image)[1].tofile(directory + 'train_image_{}.jpg'.format(order))
    with open(directory + 'train_label_{}.txt'.format(order), 'wt', encoding='utf8') as file:
        for i in range(total_row):
            for j in range(total_col):
                if i*total_col + j >= len(label):
                    file.write('\n')
                    return
                file.write(label[i*total_col + j])
            file.write('\n')


def main():
    gb_list = get_primer_gb()
    print(gb_list)
    print(len(gb_list))
    font_style_list = ['simhei.ttf', 'simkai.ttf', 'simsun.ttc', 'msyh.ttc', 'STXINWEI.TTF', 'SIMLI.TTF', 'FZSTK.TTF']
    all_char_list = []
    for style in font_style_list:
        print(style)
        optical_char_list, char_info_list = font2image(style, gb_list)
        ImageEnhance().enhance(optical_char_list)
        all_char_list.extend(zip(optical_char_list, char_info_list))
        # print(len(all_char_list))
    save_combined_image(all_char_list, group_size=(256, 256))


if __name__ == '__main__':
    begin = time.time()
    main()
    print('Total Time:', (time.time() - begin) / 60)
