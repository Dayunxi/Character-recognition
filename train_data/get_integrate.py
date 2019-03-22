# 求定积分
# 不同角度的文字的分布假设为期望为μ，方差为σ的正态分布
# 取μ=0，取3σ=max_angle 得到σ=10

import scipy.stats as stats
import numpy as np


def calc_norm(miu, sigma, lower, upper):
    return stats.norm.cdf(normalize(upper, miu, sigma)) - stats.norm.cdf(normalize(lower, miu, sigma))


def normalize(z, miu, sigma):
    return (z - miu) / sigma


def main():
    total_num_per_font_style = 30
    max_angle = 30
    angle_step = 2
    cdf_list = []
    for i in range(-max_angle, max_angle+1, angle_step):
        integer = calc_norm(0, 10, i-1, i+1)
        # print(integer*total_num_per_font_style, i-1, i+1)
        cdf_list.append(integer*total_num_per_font_style)
    # print(cdf_list)
    print(np.sum(np.ceil(cdf_list)))
    cdf_list = [int(num) for num in np.ceil(cdf_list)]
    print(cdf_list)
    print(len(cdf_list))
    pass


if __name__ == '__main__':
    main()
