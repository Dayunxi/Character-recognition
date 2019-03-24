import argparse
import sys
import tempfile

import tensorflow as tf

import load_train_data

import cv2 as cv
import numpy as np


FLAGS = None
WIDTH = 64
HEIGHT = 64
CHAR_NUM = 50
TOTAL_NUM = 9360
GROUP_SIZE = (256, 256)

label_id = {}
id_label = {}


def deepnn(top_k):
    weights = {
        'W_conv1': weight_variable([3, 3, 1, 64]),
        'W_conv2': weight_variable([3, 3, 64, 128]),
        'W_conv3': weight_variable([3, 3, 128, 256]),
        'W_conv4': weight_variable([3, 3, 256, 512]),
        'W_conv5': weight_variable([3, 3, 512, 512]),
    }
    biases = {
        'b_conv1': bias_variable([64]),
        'b_conv2': bias_variable([128]),
        'b_conv3': bias_variable([256]),
        'b_conv4': bias_variable([512]),
        'b_conv5': bias_variable([512]),
    }

    image = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH])
    with tf.name_scope('reshape'):
        x_image = tf.reshape(image, [-1, HEIGHT, WIDTH, 1])

    with tf.name_scope('conv1'):
        h_conv1 = tf.nn.relu(conv2d(x_image, weights['W_conv1']) + biases['b_conv1'])

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        h_conv2 = tf.nn.relu(conv2d(h_pool1, weights['W_conv2']) + biases['b_conv2'])

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('conv3'):
        h_conv3 = tf.nn.relu(conv2d(h_pool2, weights['W_conv3']) + biases['b_conv3'])

    with tf.name_scope('pool3'):
        h_pool3 = max_pool_2x2(h_conv3)

    # with tf.name_scope('conv4'):
    #     h_conv4 = tf.nn.relu(conv2d(h_pool3, weights['W_conv4']) + biases['b_conv4'])
    #
    # with tf.name_scope('conv5'):
    #     h_conv5 = tf.nn.relu(conv2d(h_conv4, weights['W_conv5']) + biases['b_conv5'])
    #
    # with tf.name_scope('pool5'):
    #     h_pool5 = max_pool_2x2(h_conv5)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([8 * 8 * 256, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool3, [-1, 8 * 8 * 256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, CHAR_NUM])
        b_fc2 = bias_variable([CHAR_NUM])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    probability = tf.nn.softmax(y_conv)
    return{
        'image': image,
        'y_conv': y_conv,
        'logits': probability,
        'keep_prob': keep_prob,
        'top_k': tf.nn.top_k(probability, k=top_k)
    }


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def predict(batch):

    y_ = tf.placeholder(tf.float32, [None, CHAR_NUM])
    graph = deepnn(3)

    correct_prediction = tf.equal(tf.argmax(graph['y_conv'], 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint('model/')
        saver.restore(sess, ckpt)
        images, labels = batch

        ratio = accuracy.eval(feed_dict={graph['image']: images, y_: labels, graph['keep_prob']: 1.})
        print("Ratio:", ratio)

        for i in range(len(images)):
            image = np.reshape(images[i], [-1, 64, 64])
            # image = images[i]
            result = sess.run(graph['top_k'], feed_dict={graph['image']: image, graph['keep_prob']: 1.})
            values, indexes = result
            # print(values)
            label = np.argmax(labels[i])
            print('Origin:', label, id_label[label])
            print('Predict:', indexes[0][0], id_label[indexes[0][0]], 'Prob:', values[0][0])
            for j in range(3):
                print(id_label[indexes[0][j]], ':', values[0][j])
            # print(images[i].shape)
            cv.imshow('test', np.array(images[i]*255, dtype=np.uint8))
            cv.waitKey()
    pass


def main():
    gb_list = get_primer_gb()
    global label_id, id_label
    for char, gb_id in gb_list:
        label_id[char] = gb_id
        id_label[gb_id] = char

    batch_size = 50
    batch_gen = load_train_data.get_batch('train_data/data_test/', batch_size, total_num=TOTAL_NUM,
                                          char_size=(WIDTH, HEIGHT), group_size=GROUP_SIZE, one_hot_length=CHAR_NUM)
    batch = next(batch_gen)
    predict(batch)


def get_primer_gb():
    start = 0xB0A1
    end = 0xD7FA
    gb_list = []
    gb_id = 0
    for i in range(start, end):
        if (i & 0xF0) >> 4 < 0xA or i & 0xF == 0x0 or i & 0xF == 0xF:
            continue
        character = str(i.to_bytes(length=2, byteorder='big'), 'gb2312')
        gb_list.append((character, gb_id))
        gb_id += 1
    return gb_list


if __name__ == '__main__':
    main()
