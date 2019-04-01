import matplotlib.pyplot as plt
import numpy as np


def draw(accuracy, loss, alpha):
    length = len(loss)
    x = [50*i for i in range(length)]
    h1, = plt.plot(x, accuracy, 'r', label='accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Step')
    plt.twinx()
    h2, = plt.plot(x, loss, 'b', label='loss')
    plt.ylabel('Loss')
    plt.legend(handles=(h1, h2), loc='center left')
    plt.title('Accuracy & Loss, Alpha=' + str(alpha))


def main():
    alpha_list = [0.00158, 0.001573, 0.00155, 0.0015]
    for i, alpha in enumerate(alpha_list):
        with open('accuracy_loss_{}.txt'.format(str(alpha)), 'r') as file:
            plt.subplot(2, 2, i + 1)
            line = file.readline()
            acc_list = np.array(line.split(', '), dtype=np.float32)
            line = file.readline()
            loss_list = np.array(line.split(', '), dtype=np.float32)
            draw(acc_list, loss_list, alpha)
    plt.show()


if __name__ == '__main__':
    main()
