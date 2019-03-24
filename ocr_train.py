from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
CHAR_NUM = 3355
TOTAL_NUM = 13*72*2*CHAR_NUM
GROUP_SIZE = (256, 256)


def deepnn(x):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.
    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.
    """

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

    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, HEIGHT, WIDTH, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        h_conv1 = tf.nn.relu(conv2d(x_image, weights['W_conv1']) + biases['b_conv1'])

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        h_conv2 = tf.nn.relu(conv2d(h_pool1, weights['W_conv2']) + biases['b_conv2'])

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('conv3'):
        h_conv3 = tf.nn.relu(conv2d(h_pool2, weights['W_conv3']) + biases['b_conv3'])

    with tf.name_scope('pool3'):
        h_pool3 = max_pool_2x2(h_conv3)

    with tf.name_scope('conv4'):
        h_conv4 = tf.nn.relu(conv2d(h_pool3, weights['W_conv4']) + biases['b_conv4'])

    with tf.name_scope('conv5'):
        h_conv5 = tf.nn.relu(conv2d(h_conv4, weights['W_conv5']) + biases['b_conv5'])

    with tf.name_scope('pool5'):
        h_pool5 = max_pool_2x2(h_conv5)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([4 * 4 * 512, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool5, [-1, 4 * 4 * 512])
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
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    # Import data
    # mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, CHAR_NUM])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    batch_size = 50
    batch_gen = load_train_data.get_batch(batch_size, total_num=TOTAL_NUM, group_size=GROUP_SIZE)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        try:
            for i in range(100000):
                batch = next(batch_gen)
                images, labels = batch
                # print(images[0])
                # for image in images:
                #     for row in image:
                #         print(row)
                #     cv.imshow('test', np.array(images[0]*255, dtype=np.uint8))
                #     cv.waitKey()
                # for j in range(batch_size):
                #     for pos in range(3355):
                #         if labels[j][pos] == 1:
                #             print(pos, j)
                #             break
                #     # print(labels[j])
                #     cv.imshow('test', images[j])
                #     cv.waitKey()
                if i % 10 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: images, y_: labels, keep_prob: 1.0})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                train_step.run(feed_dict={x: images, y_: labels, keep_prob: 0.5})
                if i == 1000:
                    saver.save(sess, 'model/ocr-model', global_step=i)
        except StopIteration:
            pass

        # print('test accuracy %g' % accuracy.eval(feed_dict={
        #     x: mnist.test.images[:1000], y_: mnist.test.labels[:1000], keep_prob: 1.0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)