from PIL import Image, ImageFont, ImageDraw
import cv2 as cv
import numpy as np
import os


class ImageEnhance(object):
    def __init__(self, noise=True, dilate=True, erode=True):
        self.noise = noise
        self.dilate = dilate
        self.erode = erode

    @classmethod
    def add_noise(cls, img):
        for i in range(20):     # 添加点噪声
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
            if self.dilate and np.random.random() < 0.5:
                img_list[i] = self.add_dilate(im)
            elif self.erode:
                img_list[i] = self.add_erode(im)
            if self.noise and np.random.random() < 0.5:
                img_list[i] = self.add_noise(im)
            # cv.imshow('ll,', img_list[i])
            # cv.waitKey()


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
        print(char)
        image.paste(0, (0, 0, canvas_width, canvas_height))  # erase
        draw.text((5, 5), char, font=font, fill=255)
        for i, angle in enumerate(range(-max_angle, max_angle+1, angle_step)):
            optical_char = image.rotate(angle, Image.BILINEAR)   # linear interpolation
            box = optical_char.getbbox()
            optical_char = optical_char.crop(box)
            optical_char = optical_char.resize((width, height), Image.ANTIALIAS)
            # optical_char_list.append((optical_char, char))
            # print(i, amount[i], angle)
            for _ in range(amount[i]):
                optical_char_list.append(np.array(optical_char))
                char_info_list.append(char)
            # cv.imshow(char, np.array(optical_char))
            # cv.waitKey()
    return optical_char_list, char_info_list


def save_to_local(optical_char_list, char_info_list, font_style):
    order = {}
    for image, char in zip(optical_char_list, char_info_list):
        if char not in order:
            order[char] = 1
        file_dict = 'character/{}/'.format(char)
        file_name = '{}_{}_{}.jpg'.format(char, font_style, order[char])
        # print(char)
        if not os.path.exists(file_dict):
            os.mkdir(file_dict)
        # image.save(file_dict + file_name)
        # cv.imwrite(file_dict + file_name, image)  # 不支持中文
        cv.imencode('.jpg', image)[1].tofile(file_dict+file_name)
        order[char] += 1
    pass


def image_combine(optical_char_list):
    pass


def main():
    gb_list = get_primer_gb()
    print(gb_list)
    print(len(gb_list))
    font_style_list = ['simhei.ttf', 'simkai.ttf', 'simsun.ttc', 'msyh.ttc', 'STXINWEI.TTF', 'SIMLI.TTF', 'FZSTK.TTF']
    for style in font_style_list[:]:
        print(style)
        optical_char_list, char_info_list = font2image(style, gb_list[:10])
        ImageEnhance().enhance(optical_char_list)
        save_to_local(optical_char_list, char_info_list, style.split('.')[0])


if __name__ == '__main__':
    main()
