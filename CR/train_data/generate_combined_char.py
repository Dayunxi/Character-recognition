from PIL import Image, ImageFont, ImageDraw
import cv2 as cv
import numpy as np
import time
import copy

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
        height, width = img.shape
        for i in range(noise_num):    # 椒盐噪声
            temp_x = np.random.randint(0, height)
            temp_y = np.random.randint(0, width)
            img[temp_x][temp_y] = 255

    @classmethod
    def get_erode(cls, img):
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
        img = cv.erode(img, kernel)
        return img

    @classmethod
    def add_random_erode(cls, img):
        erode_num = np.random.randint(15, 25)
        height, width = img.shape
        for i in range(erode_num):
            temp_x = np.random.randint(0, height)
            temp_y = np.random.randint(0, width)
            img[temp_x:temp_x+3, temp_y:temp_y+3] = 0

    @classmethod
    def get_dilate(cls, img):
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        img = cv.dilate(img, kernel)
        return img

    @classmethod
    def get_gaussian(cls, img):
        bg = np.random.normal(20, 5, img.shape)
        img = img + bg
        img[img < 0] = 0
        img[img > 255] = 255
        return img

    def enhance(self, img_list):
        print('Enhancing ...')
        ret_img_list = copy.deepcopy(img_list)
        length = len(ret_img_list)
        for i in range(length):
            if np.random.random() < 0.3:
                pass
            elif self.dilate and np.random.random() < 0.5:
                ret_img_list[i] = self.get_dilate(ret_img_list[i])
            elif self.erode:
                ret_img_list[i] = self.get_erode(ret_img_list[i])
            if np.random.random() < 0.5:   # 局部腐蚀
                self.add_random_erode(ret_img_list[i])
            if np.random.random() < 0.8:   # 高斯噪声
                ret_img_list[i] = self.get_gaussian(ret_img_list[i])
            self.add_noise(ret_img_list[i])  # 一定有随机白点
        print('Done')
        return ret_img_list


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


def font2image(font_style, gb_list, char_size=(64, 64), max_angle=30, angle_step=3):
    width, height = char_size
    font_size = int((width+height)/2)
    # canvas_width = int(1.5*width)       # 大一点避免旋转后被动裁剪
    # canvas_height = int(1.5*height)

    image = Image.new('L', char_size, 0)  # L=gray RGB=rgb
    # 创建Font对象:
    font = ImageFont.truetype('font/' + font_style, font_size)

    optical_char_list = []
    char_info_list = []
    num_per_angle, amount = get_integrate.get_distribution(max_angle, angle_step, scale=20)
    print('Total num of all angle per char:', amount)
    gb_list = [char for char, _ in gb_list]
    for i, char in enumerate(gb_list):
        if i % 100 == 0:
            print('[+]Font2image: {:.2f}% ...'.format(i/len(gb_list)*100))
        image.paste(0, (0, 0, width, height))  # erase
        font_w, font_h = font.getsize(char)
        max_v = max(font_w, font_h)

        font_image = Image.new('L', (font_w, font_h), 0)
        font_draw = ImageDraw.Draw(font_image)
        font_draw.text((0, 0), char, font=font, fill=255)
        if max_v > char_size[0]:
            font_image = font_image.resize((int(font_w * width / max_v), int(font_h * height / max_v)), Image.BILINEAR)
            font_w, font_h = font_image.size

        left = int((width - font_w) / 2)
        upper = int((height - font_h) / 2)
        image.paste(font_image, (left, upper))  # 粘贴至容器中央

        for i, angle in enumerate(range(-max_angle, max_angle+1, angle_step)):
            # 由于步长为3度，过于倾斜的训练集较少，被裁剪也影响不大
            rotated_char = image.rotate(angle, Image.BILINEAR)
            for _ in range(num_per_angle[i]):
                optical_char_list.append(np.array(rotated_char))
                char_info_list.append(char)
    return optical_char_list, char_info_list


def alpha2image(font_style, gb_list, char_size=(64, 64)):
    if font_style == 'STXINGKA.TTF':
        return [], []
    width, height = char_size
    font_size = int((width+height)/2)
    # canvas_width = int(1.5*width)       # 大一点避免旋转后被动裁剪
    # canvas_height = int(1.5*height)

    image = Image.new('L', char_size, 0)  # L=gray RGB=rgb
    # 创建Font对象:
    font = ImageFont.truetype('font/' + font_style, font_size)

    optical_char_list = []
    char_info_list = []

    gb_list = [char for char, _ in gb_list]
    for i, char in enumerate(gb_list):
        if i % 10 == 0:
            print('[+]Font2image: {:.2f}% ...'.format(i/len(gb_list)*100))
        image.paste(0, (0, 0, width, height))  # erase
        font_w, font_h = font.getsize(char)
        max_v = max(font_w, font_h)

        font_image = Image.new('L', (font_w, font_h), 0)
        font_draw = ImageDraw.Draw(font_image)
        font_draw.text((0, 0), char, font=font, fill=255)
        if max_v > char_size[0]:
            font_image = font_image.resize((int(font_w * width / max_v), int(font_h * height / max_v)), Image.BILINEAR)
            font_w, font_h = font_image.size

        left = int((width - font_w) / 2)
        upper = int((height - font_h) / 2)
        image.paste(font_image, (left, upper))  # 粘贴至容器中央

        extend_images = alpha_extend(image, char_size)
        for item in extend_images:
            # if char == 'I':
            #     cv.imshow('extend', item)
            #     cv.waitKey()
            optical_char_list.append(item)
            char_info_list.append(char)
    return optical_char_list, char_info_list


def punctuation2image(font_style, gb_list, char_size=(64, 64)):
    width, height = char_size
    font_size = int((width+height)/2)

    image = Image.new('L', char_size, 0)  # L=gray RGB=rgb
    # 创建Font对象:
    font = ImageFont.truetype('font/' + font_style, font_size)
    # 创建Draw对象:
    # draw = ImageDraw.Draw(image)
    optical_punc_list = []
    punc_info_list = []
    print('[+]Punctuation2image ...')
    gb_list = [char for char, _ in gb_list]
    for i, char in enumerate(gb_list):
        if i % 5 == 0:
            print('[+]Font2image: {:.2f}% ...'.format(i/len(gb_list)*100))
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
        # if font_image is None:
        #     draw.text((left, upper), char, font=font, fill=255)
        # else:
        image.paste(font_image, (left, upper))

        optical_punc_list.append(np.array(image))
        punc_info_list.append(char)
    return optical_punc_list, punc_info_list


# 九次位移，一次膨胀，一张原图
def alpha_extend(image, char_size):
    box = image.getbbox()
    pure_image = image.crop(box)
    # pure_image.show()
    width, height = pure_image.size
    w, h = char_size
    # print('size', pure_image.size)

    canvas = np.zeros(char_size, dtype=np.uint8)
    if height > width:
        new_w = width*h//height
        expand_image = pure_image.resize((new_w, h))
        bias = (w-new_w)//2
        canvas[:, bias:bias+new_w] = np.array(expand_image, dtype=np.uint8)
    else:
        new_h = height*w//width
        expand_image = pure_image.resize((w, new_h))
        bias = (h - new_h) // 2
        canvas[bias:bias+new_h, :] = np.array(expand_image, dtype=np.uint8)

    extend_ret = [np.array(image, dtype=np.uint8), canvas]

    scale = 16
    xy_list = [(w//scale, h//scale), ((w-width)//2, h//scale), ((scale-1)*w//scale - width, h//scale),
               (w//scale, (h-height)//2), ((w-width)//2, (h-height)//2), ((scale-1)*w//scale - width, (h-height)//2),
               (w//scale, (scale-1)*h//scale - height), ((w-width)//2, (scale-1)*h//scale - height), ((scale-1)*w//scale - width, (scale-1)*h//scale - height)]
    for xy in xy_list:
        x, y = xy
        # print(xy)
        canvas = np.zeros(char_size, dtype=np.uint8)
        temp = pure_image
        if x < 0:
            x = w-width
        if y < 0:
            y = h-height
        if y + height >= h:
            temp = pure_image.resize((width*(h-y)//height, h-y))
        if x + width >= w:
            temp = pure_image.resize((w-x, height*(w-x)//width))
        width, height = temp.size
        canvas[y:y+height, x:x+width] = np.array(temp, dtype=np.uint8)
        extend_ret.append(canvas)

    return extend_ret


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
    gb_alpha_list = gb_list[3755:3755+52+10]

    font_style_list = ['simhei.ttf', 'simkai.ttf', 'simsun.ttc', 'msyh.ttc', 'STXINWEI.TTF', 'SIMLI.TTF', 'FZSTK.TTF',
                       'Deng.ttf', 'STXINGKA.TTF', 'FZYTK.TTF', 'simfang.ttf', 'SIMYOU.TTF', 'STSONG.TTF']

    all_char_list = []
    for i, style in enumerate(font_style_list):
        print(style, '{}/{}'.format(i+1, len(font_style_list)))
        optical_char_list, char_info_list = font2image(style, gb_char_list, char_size=(64, 64), angle_step=3)
        optical_char_list.extend(ImageEnhance().enhance(optical_char_list))
        char_info_list *= 2
        all_char_list.extend(zip(optical_char_list, char_info_list))

        optical_alpha_list, alpha_info_list = alpha2image(style, gb_alpha_list, char_size=(64, 64))
        optical_alpha_list *= 3  # 已扩大11倍，再扩大3倍
        alpha_info_list *= 3
        enhance1 = ImageEnhance(erode=False).enhance(optical_alpha_list)
        enhance2 = ImageEnhance(erode=False).enhance(optical_alpha_list)
        optical_alpha_list.extend(enhance1 + enhance2)
        alpha_info_list *= 3
        all_char_list.extend(zip(optical_alpha_list, alpha_info_list))

        optical_punc_list, punc_info_list = punctuation2image(style, gb_punc_list, char_size=(64, 64))
        optical_punc_list *= 6          # 由于没有旋转，扩大6倍
        punc_info_list *= 6
        enhance1 = ImageEnhance(erode=False).enhance(optical_punc_list)
        enhance2 = ImageEnhance(erode=False).enhance(optical_punc_list)
        optical_punc_list.extend(enhance1 + enhance2)
        punc_info_list *= 3
        all_char_list.extend(zip(optical_punc_list, punc_info_list))

    print('Shuffle ...')
    np.random.shuffle(all_char_list)
    test_data_ratio = 0.05
    test_data_length = int(len(all_char_list)*test_data_ratio)

    path_train = 'data_train/'
    path_test = 'data_test/'
    save_combined_image(path_train, all_char_list[:-test_data_length], char_size=(64, 64), group_size=(256, 256))
    save_combined_image(path_test, all_char_list[-test_data_length:], char_size=(64, 64), group_size=(256, 256))
    print('Total char:', len(gb_list))
    print('Total Length:', len(all_char_list))
    print('Train data length:', len(all_char_list) - test_data_length)
    print('Test data length:', test_data_length)


if __name__ == '__main__':
    begin = time.time()
    main()
    print('Total Time:', (time.time() - begin) / 60)
