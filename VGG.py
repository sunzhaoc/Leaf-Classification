import tensorflow as tf
import tensorflow.contrib.slim as slim


def vgg16(images):
    """
    VGG-16网络
    :param images:
    :return:
    """
    # VGG-16网络
    # 卷积层1  卷积核个数：64    大小：3*3  步长：1
    conv_1 = slim.conv2d(images, 64, [3, 3], 1, padding='SAME', scope='conv1')
    # 卷积层2  卷积核个数：64    大小：3*3  步长：1
    conv_2 = slim.conv2d(conv_1, 64, [3, 3], 1, padding='SAME', scope='conv2')
    # 池化层1
    max_pool_1 = slim.max_pool2d(conv_2, [2, 2], [2, 2], padding='SAME')

    # 卷积层3  卷积核个数：128    大小：3*3  步长：1
    conv_3 = slim.conv2d(max_pool_1, 128, [3, 3], 1, padding='SAME', scope='conv3')
    # 卷积层4  卷积核个数：128    大小：3*3  步长：1
    conv_4 = slim.conv2d(conv_3, 128, [3, 3], 1, padding='SAME', scope='conv4')
    # 池化层2
    max_pool_2 = slim.max_pool2d(conv_4, [2, 2], [2, 2], padding='SAME')

    # 卷积层5  卷积核个数：256    大小：3*3  步长：1
    conv_5 = slim.conv2d(max_pool_2, 256, [3, 3], 1, padding='SAME', scope='conv5')
    # 卷积层6  卷积核个数：256    大小：3*3  步长：1
    conv_6 = slim.conv2d(conv_5, 256, [3, 3], 1, padding='SAME', scope='conv6')
    # 卷积层7  卷积核个数：256    大小：3*3  步长：1
    conv_7 = slim.conv2d(conv_6, 256, [3, 3], 1, padding='SAME', scope='conv7')
    # 池化层3
    max_pool_3 = slim.max_pool2d(conv_7, [2, 2], [2, 2], padding='SAME')

    # 卷积层8  卷积核个数：512    大小：3*3  步长：1
    conv_8 = slim.conv2d(max_pool_3, 512, [3, 3], 1, padding='SAME', scope='conv8')
    # 卷积层9  卷积核个数：512    大小：3*3  步长：1
    conv_9 = slim.conv2d(conv_8, 512, [3, 3], 1, padding='SAME', scope='conv9')
    # 卷积层10  卷积核个数：512    大小：3*3  步长：1
    conv_10 = slim.conv2d(conv_9, 512, [3, 3], 1, padding='SAME', scope='conv10')
    # 池化层4
    max_pool_4 = slim.max_pool2d(conv_10, [2, 2], [2, 2], padding='SAME')

    # 卷积层11  卷积核个数：512    大小：3*3  步长：1
    conv_11 = slim.conv2d(max_pool_4, 512, [3, 3], 1, padding='SAME', scope='conv11')
    # 卷积层12  卷积核个数：512    大小：3*3  步长：1
    conv_12 = slim.conv2d(conv_11, 512, [3, 3], 1, padding='SAME', scope='conv12')
    # 卷积层13  卷积核个数：512    大小：3*3  步长：1
    conv_13 = slim.conv2d(conv_12, 512, [3, 3], 1, padding='SAME', scope='conv13')
    # 池化层5
    max_pool_5 = slim.max_pool2d(conv_13, [2, 2], [2, 2], padding='SAME')

    flatten = slim.flatten(max_pool_5)

    fc1 = slim.fully_connected(slim.dropout(flatten, 0.5), 4096, activation_fn=tf.nn.relu, scope='fc1')
    fc2 = slim.fully_connected(slim.dropout(fc1, 0.5), 4096, activation_fn=tf.nn.relu, scope='fc2')
    fc3 = slim.fully_connected(fc2, 1000, activation_fn=None, scope='fc3')

    return fc3