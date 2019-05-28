import tensorflow as tf


def alexnet(images):
    w_conv1 = tf.Variable(tf.truncated_normal([11, 11, 3, 96], stddev=0.1))
    x = tf.nn.conv2d(images, w_conv1, strides=[1, 4, 4, 1], padding='SAME')
    x = tf.nn.relu(x)

    x = tf.nn.lrn(x)

    x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    w_conv1 = tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=0.1))
    x = tf.nn.conv2d(x, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.relu(x)

    x = tf.nn.lrn(x)

    x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    w_conv1 = tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=0.1))
    x = tf.nn.conv2d(x, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.relu(x)

    w_conv1 = tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=0.1))
    x = tf.nn.conv2d(x, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.relu(x)

    w_conv1 = tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=0.1))
    x = tf.nn.conv2d(x, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.relu(x)

    x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, units=4096, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=4096, activation=tf.nn.relu)

    x = tf.layers.dense(x, units=1000, activation=None)

    return x












