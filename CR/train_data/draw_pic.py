import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# myfont = matplotlib.font_manager.FontProperties(fname="font/Deng.ttf", size=16)
plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def draw(accuracy, loss, alpha):
    length = len(loss)
    x = [25*i for i in range(length)]
    h1, = plt.plot(x, accuracy, 'r', label='准确率')
    plt.ylabel('准确率')
    plt.xlabel('步数')
    plt.twinx()
    h2, = plt.plot(x, loss, 'b', label='损失')
    plt.ylabel('损失')
    plt.legend(handles=(h1, h2), loc='center right')
    plt.title('准确率与损失, LR=' + str(alpha))


def main():
    alpha_list = [0.00158]
    for i, alpha in enumerate(alpha_list):
        with open('accuracy_loss_{}.txt'.format(str(alpha)), 'r') as file:
            plt.subplot(1, 1, 0 + 1)
            line = file.readline()
            acc_list = np.array(line.split(', '), dtype=np.float32)
            line = file.readline()
            loss_list = np.array(line.split(', '), dtype=np.float32)
            print(np.average(acc_list[-100:]))
            print(np.average(loss_list[-100:]))
            draw(acc_list, loss_list, alpha)
    plt.show()


if __name__ == '__main__':
    main()
