# 求定积分
# 不同角度的文字的分布假设为期望为μ，方差为σ的正态分布
# 取μ=0，取3σ=max_angle 得到σ=10

import scipy.stats as stats
import numpy as np


def calc_norm(miu, sigma, lower, upper):
    return stats.norm.cdf(normalize(upper, miu, sigma)) - stats.norm.cdf(normalize(lower, miu, sigma))


def normalize(z, miu, sigma):
    return (z - miu) / sigma


def get_distribution(max_angle, angle_step, scale=30, uniform=False):
    if uniform:
        length = int(max_angle*2/angle_step)+1
        return np.ones(length, dtype=np.int), length

    cdf_list = []
    for i in range(-max_angle, max_angle+1, angle_step):
        integer = calc_norm(0, 10, i - angle_step/2, i + angle_step/2)
        cdf_list.append(integer * scale)
    distribution = np.asarray(np.ceil(cdf_list), dtype=np.int)
    return distribution, np.sum(distribution)


def main():
    scale = 30
    max_angle = 30
    angle_step = 1
    cdf_list = []
    for i in range(-max_angle, max_angle+1, angle_step):
        integer = calc_norm(0, 10, i-angle_step/2, i+angle_step/2)
        # print(integer*total_num_per_font_style, i-1, i+1)
        cdf_list.append(integer*scale)
    print(cdf_list)
    print(np.sum(cdf_list))
    print(np.sum(np.ceil(cdf_list)))
    cdf_list = [int(num) for num in np.ceil(cdf_list)]
    print(cdf_list)
    print(len(cdf_list))
    pass


if __name__ == '__main__':
    print(get_distribution(30, 1))
    print(get_distribution(30, 1, uniform=True))
    # main()
