from PIL import Image, ImageFont, ImageDraw
import cv2 as cv
import numpy as np


def get_primer_gb(gb_list):
    start = 0xB0A1
    end = 0xD7FA
    for i in range(start, end):
        if (i & 0xF0) >> 4 < 0xA or i & 0xF == 0x0 or i & 0xF == 0xF:
            continue
        character = str(i.to_bytes(length=2, byteorder='big'), 'gb2312')
        gb_list.append(character)
        # print(character, i)


def font2image(font_style, char_list):
    font_size = 32
    # width = int(font_size * np.sqrt(2))  # 45 degree
    width = int(font_size * (np.sqrt(3) + 1) / 2)  # 30 degree
    height = width
    bias = int((height-font_size)/2)
    max_angle = 30
    image = Image.new('L', (width, height), 0)  # L=gray RGB=rgb
    # 创建Font对象:
    font = ImageFont.truetype(font_style, font_size)
    # 创建Draw对象:
    draw = ImageDraw.Draw(image)

    # 输出文字:
    draw.text((bias, bias), '啊', font=font, fill=255)
    image = image.rotate(max_angle)
    box = image.getbbox()
    image = image.crop(box)
    # image.save('test.jpg', 'jpeg')
    cv.imshow('cv', np.array(image))
    cv.waitKey()
    pass


# print(str(0xb1a0.to_bytes(length=2, byteorder='big'), 'gb2312'))
def main():
    # gb_list = []
    # get_primer_gb(gb_list)
    # print(gb_list)
    # print(len(gb_list))
    font2image('simhei.ttf', 0)


if __name__ == '__main__':
    main()
