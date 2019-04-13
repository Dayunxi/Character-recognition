import tensorflow as tf
import time

import load_train_data

import numpy as np


FLAGS = None
WIDTH = 64
HEIGHT = 64
CHAR_NUM = 3859
TOTAL_NUM = 65532   # int(((CHAR_NUM-42)*13*33*2 + 42*13*3)*0.98)
BATCH_SIZE = 128
GROUP_SIZE = (256, 256)
ALPHA = 0.00158
STDDEV = 0.085
INFO_STEP = 10
SAVE_STEP = 10000

label_id = {}
id_label = {}


def deepnn(top_k):
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

    image = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH])

    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(image, [-1, HEIGHT, WIDTH, 1])

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

    with tf.name_scope('pool4'):
        h_pool4 = max_pool_2x2(h_conv5)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        h_pool4_flat = tf.reshape(h_pool4, [-1, 4 * 4 * 512])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, weights['W_fc1']) + biases['b_fc1'])

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        y_conv = tf.matmul(h_fc1_drop, weights['W_fc2']) + biases['b_fc2']

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


def test():

    y_ = tf.placeholder(tf.float32, [None, CHAR_NUM])
    graph = deepnn(3)

    correct_prediction = tf.equal(tf.argmax(graph['y_conv'], 1), tf.argmax(y_, axis=1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=graph['y_conv'])
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.Session() as sess:
        path_train = 'train_data/data_test/'
        batch_gen = load_train_data.get_batch(path_train, BATCH_SIZE, total_num=TOTAL_NUM, one_hot_length=CHAR_NUM,
                                              char_size=(WIDTH, HEIGHT), group_size=GROUP_SIZE)

        saver = tf.train.Saver()
        print('Loading model')
        ckpt_path = 'model/ocr-model-25138'
        saver.restore(sess, ckpt_path)
        print('Load Done')

        time_begin = time.time()
        accuracy_list = []
        loss_list = []
        curr_step = 1
        try:
            while True:
                batch = next(batch_gen)
                images, labels = batch
                test_accuracy = accuracy.eval(feed_dict={graph['image']: images, y_: labels, graph['keep_prob']: 1.0})
                test_loss = cross_entropy.eval(feed_dict={graph['image']: images, y_: labels, graph['keep_prob']: 1.0})

                accuracy_list.append(test_accuracy)
                loss_list.append(test_loss)
                if curr_step % INFO_STEP == 0:
                    print('Step {}, Total Time {:.1f}min, Training accuracy {:.2f}%, Loss {}'.format(
                        curr_step, (time.time() - time_begin) / 60, test_accuracy * 100, test_loss))
                    print('Current average accuracy: {:.2f}%, average loss: {:.5f}\n'.format(
                        np.average(accuracy_list)*100, np.average(loss_list)))
                curr_step += 1
        except StopIteration:
            save_acc_loss(accuracy_list, loss_list)
            print('Done')
        except KeyboardInterrupt:
            if loss_list[-1] < 1:
                save_acc_loss(accuracy_list, loss_list)


def save_acc_loss(acc, loss):
    with open('test_accuracy_loss_{}.txt'.format(ALPHA), 'wt') as file:
        file.write(', '.join([str(num) for num in acc]))
        file.write('\n')
        file.write(', '.join([str(num) for num in loss]))


def main():
    # gb_list = get_chinese.get_primary_gb()
    # global label_id, id_label
    # for char, gb_id in gb_list:
    #     label_id[char] = gb_id
    #     id_label[gb_id] = char
    test()


if __name__ == '__main__':
    main()
