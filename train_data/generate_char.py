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


def save_single_image(optical_char_list, char_info_list, font_style):
    order = {}
    for image, char in zip(optical_char_list, char_info_list):
        if char not in order:
            order[char] = 1

        file_dict = 'character/{}/'.format(char)
        file_name = '{}_{}_{}.jpg'.format(char, font_style, order[char])
        if os.path.exists(file_dict + file_name):
            continue
        if not os.path.exists(file_dict):
            os.mkdir(file_dict)
        # cv.imwrite(file_dict + file_name, image)  # 不支持中文
        cv.imencode('.jpg', image)[1].tofile(file_dict+file_name)
        order[char] += 1
    pass


def save_combined_image(optical_char_list, char_info_list, font_style):
    word_dict = {}
    for image, char in zip(optical_char_list, char_info_list):
        if char not in word_dict:
            word_dict[char] = []
        word_dict[char].append(image)
    for order, key in enumerate(word_dict):
        if order % 100 == 0:
            print('[+]Save: ' + str(order/len(word_dict)*100) + '% ...')
        file_dict = 'character/{}/'.format(key)
        file_name = '{}_{}.jpg'.format(key, font_style)
        if os.path.exists(file_dict + file_name):
            # print('覆盖:', file_name)
            continue
        big_canvas = np.zeros((32, 32 * 49), dtype=np.uint8)
        for i, image in enumerate(word_dict[key]):
            big_canvas[:, 32*i:32*(i+1)] = image
        if not os.path.exists(file_dict):
            os.mkdir(file_dict)
        cv.imencode('.jpg', big_canvas)[1].tofile(file_dict + file_name)
        # cv.imshow('test', big_canvas)
        # cv.waitKey()


def main():
    gb_list = get_primer_gb()
    print(gb_list)
    print(len(gb_list))
    font_style_list = ['simhei.ttf', 'simkai.ttf', 'simsun.ttc', 'msyh.ttc', 'STXINWEI.TTF', 'SIMLI.TTF', 'FZSTK.TTF']
    for style in font_style_list:
        print(style)
        optical_char_list, char_info_list = font2image(style, [('一', 1), ('二', 2), ('三', 3)])
        ImageEnhance().enhance(optical_char_list)
        save_combined_image(optical_char_list, char_info_list, style.split('.')[0])
        # save_single_image(optical_char_list, char_info_list, style.split('.')[0])


if __name__ == '__main__':
    start = time.time()
    main()
    print('Total Time:', (time.time() - start) / 60)
