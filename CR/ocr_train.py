from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

import tensorflow as tf

import load_train_data

import time


FLAGS = None
WIDTH = 64
HEIGHT = 64
CHAR_NUM = 3859
TOTAL_NUM = 3120574
BATCH_SIZE = 128
GROUP_SIZE = (256, 256)
ALPHA = 0.00158
STDDEV = 0.085
INFO_STEP = 25
SAVE_STEP = 10000


def deepnn(x):
    """构建网络结构
    """

    weights = {
        'W_conv1': weight_variable([3, 3, 1, 64]),
        'W_conv2': weight_variable([3, 3, 64, 128]),
        'W_conv3': weight_variable([3, 3, 128, 256]),
        'W_conv4': weight_variable([3, 3, 256, 512]),
        'W_conv5': weight_variable([3, 3, 512, 512]),
        'W_fc1': weight_variable([4 * 4 * 512, 2048]),
        'W_fc2': weight_variable([2048, CHAR_NUM])
    }
    biases = {
        'b_conv1': bias_variable([64]),
        'b_conv2': bias_variable([128]),
        'b_conv3': bias_variable([256]),
        'b_conv4': bias_variable([512]),
        'b_conv5': bias_variable([512]),
        'b_fc1': bias_variable([2048]),
        'b_fc2': bias_variable([CHAR_NUM])
    }

    # 调整输入为四维矩阵
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, HEIGHT, WIDTH, 1])

    # 第一层卷积，将灰度图映射为64个特征图
    with tf.name_scope('conv1'):
        h_conv1 = tf.nn.relu(conv2d(x_image, weights['W_conv1']) + biases['b_conv1'])

    # 第一层池化，缩小两倍
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # 第二次卷积，64个特征图映射为128个特征图
    with tf.name_scope('conv2'):
        h_conv2 = tf.nn.relu(conv2d(h_pool1, weights['W_conv2']) + biases['b_conv2'])

    # 第二层池化，缩小两倍
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # 第三次卷积，128个特征图映射为256个特征图
    with tf.name_scope('conv3'):
        h_conv3 = tf.nn.relu(conv2d(h_pool2, weights['W_conv3']) + biases['b_conv3'])

    # 第三层池化，缩小两倍
    with tf.name_scope('pool3'):
        h_pool3 = max_pool_2x2(h_conv3)
    # 第四次卷积，256个特征图映射为512个特征图
    with tf.name_scope('conv4'):
        h_conv4 = tf.nn.relu(conv2d(h_pool3, weights['W_conv4']) + biases['b_conv4'])
    # 第五次卷积，512个特征图映射为512个特征图
    with tf.name_scope('conv5'):
        h_conv5 = tf.nn.relu(conv2d(h_conv4, weights['W_conv5']) + biases['b_conv5'])

    # 第四层池化，缩小两倍
    with tf.name_scope('pool4'):
        h_pool4 = max_pool_2x2(h_conv5)

    # 经过四次降采样，图片由64x64变为4x4
    with tf.name_scope('fc1'):
        h_pool4_flat = tf.reshape(h_pool4, [-1, 4 * 4 * 512])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, weights['W_fc1']) + biases['b_fc1'])

    # 在第一层全连接使用dropout防止过拟合
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 将2048个特征值映射为3859个类别
    with tf.name_scope('fc2'):
        y_conv = tf.matmul(h_fc1_drop, weights['W_fc2']) + biases['b_fc2']
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=STDDEV)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial)


def main():
    time_begin = time.time()

    x = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH])

    y_ = tf.placeholder(tf.float32, [None, CHAR_NUM])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(ALPHA).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    print('Alpha:', ALPHA)

    accuracy_list = []
    loss_list = []

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        path_train = 'train_data/data_train/'
        batch_gen = load_train_data.get_batch(path_train, BATCH_SIZE, total_num=TOTAL_NUM, one_hot_length=CHAR_NUM,
                                              char_size=(WIDTH, HEIGHT), group_size=GROUP_SIZE)
        curr_step = 1
        try:
            while True:
                batch = next(batch_gen)
                images, labels = batch
                if curr_step % INFO_STEP == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: images, y_: labels, keep_prob: 1.0})
                    train_loss = cross_entropy.eval(feed_dict={x: images, y_: labels, keep_prob: 1.0})
                    print('Step {}, Total Time {:.1f}min, Training accuracy {:.2f}%, Loss {}'.format(
                        curr_step, (time.time()-time_begin)/60, train_accuracy*100, train_loss))
                    accuracy_list.append(train_accuracy)
                    loss_list.append(train_loss)
                train_step.run(feed_dict={x: images, y_: labels, keep_prob: 0.5})
                if curr_step % SAVE_STEP == 0:
                    saver.save(sess, 'model/ocr-model', global_step=curr_step)
                curr_step += 1
        except StopIteration:  # 读取所有训练集后保存模型
            save_acc_loss(accuracy_list, loss_list)
            saver.save(sess, 'model/ocr-model', global_step=curr_step)
        except KeyboardInterrupt:
            if loss_list[-1] < 1:
                save_acc_loss(accuracy_list, loss_list)
                saver.save(sess, 'model/ocr-model', global_step=curr_step)
    display_config()


def display_config():
    print('char num:', CHAR_NUM)
    print('learning rate:', ALPHA)
    print('stddev:', STDDEV)
    print('batch size:', BATCH_SIZE)


def save_acc_loss(acc, loss):
    with open('accuracy_loss_{}.txt'.format(ALPHA), 'wt') as file:
        file.write(', '.join([str(num) for num in acc]))
        file.write('\n')
        file.write(', '.join([str(num) for num in loss]))


if __name__ == '__main__':
    main()
