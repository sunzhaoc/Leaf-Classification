import logging
import os
import pickle
import random
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image

logger = logging.getLogger('Leaf Classification')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

# 图像预处理
tf.app.flags.DEFINE_boolean('random_flip_up_down', True, "上下翻转")
tf.app.flags.DEFINE_boolean('random_flip_left_right', True, "左右翻转")
tf.app.flags.DEFINE_boolean('random_brightness', True, "图片亮度")
tf.app.flags.DEFINE_boolean('random_contrast', True, "图片对比度")
tf.app.flags.DEFINE_boolean('random_saturation', False, "图片饱和度")
tf.app.flags.DEFINE_boolean('random_hue', False, "图片饱和度")
tf.app.flags.DEFINE_boolean('resize_image_with_crop_or_pad', False, "随机裁剪")

tf.app.flags.DEFINE_integer('charset_size', 6666,
                            "Choose the first `charset_size` character to conduct our experiment.")
tf.app.flags.DEFINE_integer('image_size', 224, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_integer('pic_channel', 3, "彩色图片")
# tf.app.flags.DEFINE_integer('pic_channel', 1, "灰度图片")
tf.app.flags.DEFINE_integer('max_steps', 10000, '最大训练步数 ')
tf.app.flags.DEFINE_integer('eval_steps', 2, "显示步数")
tf.app.flags.DEFINE_integer('save_steps', 500, "保存步数")

tf.app.flags.DEFINE_string('checkpoint_dir', './model/RESNET50_100%/', '模型保存路径')
tf.app.flags.DEFINE_string('train_data_dir', './data/', '训练集目录')
tf.app.flags.DEFINE_string('test_data_dir', './test/', '测试集目录')
tf.app.flags.DEFINE_string('log_dir', './log/RESNET50_100%/', '日志路径')
tf.app.flags.DEFINE_string('test_dir', './test/00016/16001.jpg', 'test模式测试图片的路径')

tf.app.flags.DEFINE_integer('axis', 1, 'axis轴')
tf.app.flags.DEFINE_boolean('train', True, 'Resnet50是否在训练（还是在测试）')
tf.app.flags.DEFINE_boolean('restore', False, '是否重载模型')
tf.app.flags.DEFINE_integer('epoch', 10, 'Number of epoches')
tf.app.flags.DEFINE_integer('batch_size', 50, '验证批次大小V')
tf.app.flags.DEFINE_string('mode', 'train', '训练的模式：train valid test"}')
FLAGS = tf.app.flags.FLAGS


class DataIterator:
    def __init__(self, data_dir):
        # 如果可用的计算力不够，将FLAGS.charset_size的值设小
        # 获取图片路径 = 主目录+6位分类目录
        truncate_path = data_dir + ('%05d' % FLAGS.charset_size)
        # print(truncate_path)
        # print(len(truncate_path))

        self.image_names = []
        for root, sub_folder, file_list in os.walk(data_dir):
            print(root)
            if root < truncate_path:  # eg. root='train/00000' truncate_path='../data/train/01000'
                self.image_names += [os.path.join(root, file_path) for file_path in file_list]

        # 打乱所有图片顺序（整个训练图库下的文件）
        # print(self.image_names)
        random.shuffle(self.image_names)
        print(self.image_names)

        # 取label
        self.labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.image_names]
        print(self.labels)

    @property
    def size(self):
        return len(self.labels)

    @staticmethod
    def data_augmentation(images):
        """
        图像预处理
        :param images:
        :return:
        """
        # 随机上下翻转
        if FLAGS.random_flip_up_down:
            images = tf.image.random_flip_up_down(images)

        # 随机左右翻转
        if FLAGS.random_flip_left_right:
            images = tf.image.random_flip_left_right(images)

        # 在某范围随机调整图片亮度
        if FLAGS.random_brightness:
            images = tf.image.random_brightness(images, max_delta=0.1)

        # 在某范围随机调整图片对比度
        if FLAGS.random_contrast:
            images = tf.image.random_contrast(images, 0.9, 1.1)

        # 随机裁剪
        if FLAGS.resize_image_with_crop_or_pad:
            images = tf.image.resize_image_with_crop_or_pad(images, FLAGS.image_size, FLAGS.image_size)

        # 随机饱和度
        if FLAGS.random_saturation:
            images = tf.image.random_saturation(images, 0.9, 1.1)

        # 随机色调
        if FLAGS.random_hue:
            images = tf.image.random_hue(images, max_delta=0.1)

        return images

    def input_pipeline(self, batch_size, num_epochs=None):
        """

        :param batch_size:
        :param num_epochs:一个整数，可调。
        如果指定，slice_input_producer将在生成OutOfRange错误之前遍历num_epochs次。
        如果没有指定，slice_input_producer可以无限次循环遍历。
        :return:
        """
        # 将图片名和label转换成tensorflow可用的tensor
        images_tensor = tf.convert_to_tensor(self.image_names, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)

        # tensor生成器，默认shuffle=True样本数据打乱, seed=None随机数种子不固定随机水平
        input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor], num_epochs=num_epochs)
        # 取标签
        labels = input_queue[1]
        # 读取图片
        images_content = tf.read_file(input_queue[0])
        # 解码图片，图片归一化
        images = tf.image.convert_image_dtype(tf.image.decode_jpeg(images_content, channels=3), tf.float32)

        # 调用图像预处理
        images = self.data_augmentation(images)

        # 创建新的常量
        new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
        # 调整图片大小
        images = tf.image.resize_images(images, new_size)

        # 批处理
        # min_after_dequeue: 队列中元素的最小数量，用于确保元素的混合程度
        image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=150,
                                                          min_after_dequeue=10)
        return image_batch, label_batch


def identity_block(x_input, in_filter, out_filters, stage, block):
    """
    Identity block 定义

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    training -- train or test 训练或者测试

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    block_name = 'res' + str(stage) + block
    f1, f2, f3 = out_filters
    with tf.variable_scope(block_name):
        x_shortcut = x_input

        # first
        w_conv1 = weight_variable([1, 1, in_filter, f1])
        x = tf.nn.conv2d(x_input, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.layers.batch_normalization(x, axis=FLAGS.axis, training=FLAGS.train)
        x = tf.nn.relu(x)

        # second
        w_conv2 = weight_variable([3, 3, f1, f2])
        x = tf.nn.conv2d(x, w_conv2, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.layers.batch_normalization(x, axis=FLAGS.axis, training=FLAGS.train)
        x = tf.nn.relu(x)

        # third
        w_conv3 = weight_variable([1, 1, f2, f3])
        x = tf.nn.conv2d(x, w_conv3, strides=[1, 1, 1, 1], padding='VALID')
        x = tf.layers.batch_normalization(x, axis=FLAGS.axis, training=FLAGS.train)

        # final step
        add = tf.add(x, x_shortcut)
        add_result = tf.nn.relu(add)

    return add_result


def convolutional_block(x_input, in_filter, out_filters, stage, block, stride=2):
    """
    Convolutional block 定义

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    training -- train or test
    stride -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    block_name = 'res' + str(stage) + block
    with tf.variable_scope(block_name):
        f1, f2, f3 = out_filters

        x_shortcut = x_input

        # first
        w_conv1 = weight_variable([1, 1, in_filter, f1])
        x = tf.nn.conv2d(x_input, w_conv1, strides=[1, stride, stride, 1], padding='VALID')
        x = tf.layers.batch_normalization(x, axis=FLAGS.axis, training=FLAGS.train)
        x = tf.nn.relu(x)

        # second
        w_conv2 = weight_variable([3, 3, f1, f2])
        x = tf.nn.conv2d(x, w_conv2, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.layers.batch_normalization(x, axis=FLAGS.axis, training=FLAGS.train)
        x = tf.nn.relu(x)

        # third
        w_conv3 = weight_variable([1, 1, f2, f3])
        x = tf.nn.conv2d(x, w_conv3, strides=[1, 1, 1, 1], padding='VALID')
        x = tf.layers.batch_normalization(x, axis=FLAGS.axis, training=FLAGS.train)

        # shortcut path
        w_shortcut = weight_variable([1, 1, in_filter, f3])
        x_shortcut = tf.nn.conv2d(x_shortcut, w_shortcut, strides=[1, stride, stride, 1], padding='VALID')
        x_shortcut = tf.layers.batch_normalization(x_shortcut, axis=FLAGS.axis, training=FLAGS.train)

        # final
        add = tf.add(x_shortcut, x)
        add_result = tf.nn.relu(add)

    return add_result


def resnet50(x_input):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:

    Returns:
    """
    # 填充张量
    # x = tf.pad(x_input, tf.constant([[0, 0], [3, 3, ], [3, 3], [0, 0]]), "CONSTANT")

    # with tf.variable_scope('reference'):
    # training = tf.placeholder(tf.bool, name='training')

    # stage 1
    # filter：7*7
    # input：3 channels
    # output：64 channels
    # stride: 2
    with tf.variable_scope('conv1'):
        w_conv1 = weight_variable([7, 7, 3, 64])
        x = tf.nn.conv2d(x_input, w_conv1, strides=[1, 2, 2, 1], padding='VALID')
        x = tf.layers.batch_normalization(x, axis=FLAGS.axis, training=FLAGS.train)
        x = tf.nn.relu(x)

    with tf.variable_scope("conv2_x"):
        # filter: 3*3
        # stride: 2
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        # 判定是否是这个shape
        # assert (x.get_shape() == (x.get_shape()[0], 15, 15, 64))

        # stage 2
        x = convolutional_block(x, 64, [64, 64, 256], stage=2, block='a', stride=1)
        x = identity_block(x, 256, [64, 64, 256], stage=2, block='b')
        x = identity_block(x, 256, [64, 64, 256], stage=2, block='c')

    with tf.variable_scope('conv3_x'):
        # stage 3
        x = convolutional_block(x, 256, [128, 128, 512], stage=3, block='a')
        x = identity_block(x, 512, [128, 128, 512], stage=3, block='b')
        x = identity_block(x, 512, [128, 128, 512], stage=3, block='c')
        x = identity_block(x, 512, [128, 128, 512], stage=3, block='d')

    with tf.variable_scope('conv4_x'):
        # stage 4
        x = convolutional_block(x, 512, [256, 256, 1024], stage=4, block='a')
        x = identity_block(x, 1024, [256, 256, 1024], stage=4, block='b')
        x = identity_block(x, 1024, [256, 256, 1024], stage=4, block='c')
        x = identity_block(x, 1024, [256, 256, 1024], stage=4, block='d')
        x = identity_block(x, 1024, [256, 256, 1024], stage=4, block='e')
        x = identity_block(x, 1024, [256, 256, 1024], stage=4, block='f')

    with tf.variable_scope('conv5_x'):
        # stage 5
        x = convolutional_block(x, 1024, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 2048, [512, 512, 2048], stage=5, block='b')
        x = identity_block(x, 2048, [512, 512, 2048], stage=5, block='c')

    # filter: 7*7
    # stride: 1
    x = tf.nn.avg_pool(x, [1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID')

    flatten = tf.layers.flatten(x)

    fc1 = tf.layers.dense(flatten, units=1000, activation=None)

    return fc1


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


def build_graph(top_k):
    """
    构建网络
    :param top_k:最高的几个准确率
    :return:
    """
    with tf.device('/gpu:0'):
        keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
        images = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, FLAGS.pic_channel],
                                name='image_batch')
        labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')

        # 使用的网络
        # Resnet50
        # features = tf.placeholder(tf.float32, [None, 64, 64, 3], name='feature')
        logits = resnet50(images)

    with tf.device('/gpu:0'):
        with tf.name_scope("loss"):
            # 计算损失
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            # loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels))

    with tf.device('/gpu:0'):
        with tf.name_scope("accuracy"):
            # 计算准确率
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))

        # 常量初始化函数
        global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)

        # 学习率指数衰减法
        rate = tf.train.exponential_decay(2e-4, global_step, decay_steps=2000, decay_rate=0.97, staircase=True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss, global_step=global_step)

    with tf.name_scope("probabilities"):
        with tf.device('/gpu:0'):
            # ==========================================================================================================
            # 以下是计算可能性
            probabilities = tf.nn.softmax(logits)

        # 返回probabilities中每行最大的k个数，以及它们的索引
        predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
        accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))
        # ==============================================================================================================

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('top_k', accuracy_in_top_k)

    merged_summary_op = tf.summary.merge_all()

    return {'images': images,
            'labels': labels,
            'keep_prob': keep_prob,
            'top_k': top_k,
            'global_step': global_step,
            'train_op': train_op,
            'loss': loss,
            'accuracy': accuracy,
            'accuracy_top_k': accuracy_in_top_k,
            'merged_summary_op': merged_summary_op,
            'predicted_distribution': probabilities,
            'predicted_index_top_k': predicted_index_top_k,
            'predicted_val_top_k': predicted_val_top_k}


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def train():
    print('Begin training')
    train_feeder = DataIterator(data_dir=FLAGS.train_data_dir)
    test_feeder = DataIterator(data_dir=FLAGS.test_data_dir)

    # # GPU资源设定
    # config = tf.ConfigProto(allow_soft_placement=True)
    # # 最多占gpu资源的95%
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    # # 开始不会给tensorflow全部gpu资源 而是按需增加
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)

    # 开启会话
    with tf.Session() as sess:
        train_images, train_labels = train_feeder.input_pipeline(batch_size=FLAGS.batch_size, num_epochs=FLAGS.epoch)
        test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size)

        graph = build_graph(top_k=5)

        # 初始化变量
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # 开启线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 创建模型保存和加载
        saver = tf.train.Saver()

        # 数据可视化
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

        start_step = 0

        if FLAGS.restore:
            # 自动获取最后一次保存的模型
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                # 恢复模型
                saver.restore(sess, ckpt)

                print("restore from the checkpoint {0}".format(ckpt))
                start_step += int(ckpt.split('-')[-1])

        logger.info(':::Training Start:::')

        # 异常处理部分
        try:
            while not coord.should_stop():
                # 在epoch后以秒为单位返回浮点数时间
                # Beginning time
                start_time = time.time()
                train_images_batch, train_labels_batch = sess.run([train_images, train_labels])
                feed_dict = {graph['images']: train_images_batch,
                             graph['labels']: train_labels_batch,
                             graph['keep_prob']: 0.8}
                _, loss_val, accuracy_train, train_summary, step = sess.run([graph['train_op'],
                                                                             graph['loss'],
                                                                             graph['accuracy'],
                                                                             graph['merged_summary_op'],
                                                                             graph['global_step']], feed_dict=feed_dict)
                train_writer.add_summary(train_summary, step)
                # print(train_labels_batch)
                # Ending time
                end_time = time.time()

                logger.info("the step: {0} takes {1}s   loss: {2}   accuracy: {3}%".format(round(step, 0),
                                                                                           round(end_time - start_time,
                                                                                                 2), round(loss_val, 2),
                                                                                           round(accuracy_train * 100,
                                                                                                 2)))
                if step > FLAGS.max_steps:
                    break

                if step % FLAGS.eval_steps == 1:
                    test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                    feed_dict = {graph['images']: test_images_batch,
                                 graph['labels']: test_labels_batch,
                                 graph['keep_prob']: 1.0}
                    accuracy_test, test_summary = sess.run([graph['accuracy'],
                                                            graph['merged_summary_op']], feed_dict=feed_dict)
                    test_writer.add_summary(test_summary, step)

                    logger.info('======================= Eval a batch =======================')
                    logger.info('the step: {0} test accuracy: {1} %'.format(step, round(accuracy_test * 100, 2)))
                    logger.info('======================= Eval a batch =======================')

                # Save model
                if step % FLAGS.save_steps == 1:
                    logger.info('Save the ckpt of {0}'.format(step))
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'), global_step=graph['global_step'])

        # 当操作迭代超过有效输入范围时引发
        except tf.errors.OutOfRangeError:  # Raised when an operation iterates past the valid input range.
            logger.info('==================Train Finished================')
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'), global_step=graph['global_step'])

        # 回收线程
        finally:
            coord.request_stop()
        coord.join(threads)


def validation():
    print('分类')
    test_feeder = DataIterator(data_dir=FLAGS.test_data_dir)

    final_predict_val = []
    final_predict_index = []
    groundtruth = []

    with tf.Session() as sess:
        test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size, num_epochs=1)

        graph = build_graph(top_k=1)

        # 初始化变量
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # 开启线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 加载模型
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print("restore from the checkpoint {0}".format(ckpt))
        logger.info(':::Start validation:::')

        try:
            i = 0
            acc_top_1, acc_top_k = 0.0, 0.0

            while not coord.should_stop():
                i += 1
                start_time = time.time()
                test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                feed_dict = {graph['images']: test_images_batch,
                             graph['labels']: test_labels_batch,
                             graph['keep_prob']: 1.0}

                batch_labels, probs, indices, acc_1, acc_k = sess.run([graph['labels'],
                                                                       graph['predicted_val_top_k'],
                                                                       graph['predicted_index_top_k'],
                                                                       graph['accuracy'],
                                                                       graph['accuracy_top_k']], feed_dict=feed_dict)
                final_predict_val += probs.tolist()
                final_predict_index += indices.tolist()  # Return the array as a (possibly nested) list.
                groundtruth += batch_labels.tolist()
                acc_top_1 += acc_1
                acc_top_k += acc_k
                end_time = time.time()

                logger.info("the batch {0} takes {1} seconds, accuracy = {2}(top_1) {3}(top_k) {4}"
                            .format(i, end_time - start_time, acc_1, acc_k, indices))

                # logger.info("{0}".format(indices))

        except tf.errors.OutOfRangeError:
            logger.info('==================分类结束 Validation Finished================')
            acc_top_1 = acc_top_1 * FLAGS.batch_size / test_feeder.size  # calculate the mean average
            acc_top_k = acc_top_k * FLAGS.batch_size / test_feeder.size
            logger.info('top 1 accuracy: {0}%    top k accuracy: {1}%'.format(round(acc_top_1 * 100, 2),
                                                                              round(acc_top_k * 100, 2)))

        finally:
            coord.request_stop()
        coord.join(threads)
    return {'prob': final_predict_val, 'indices': final_predict_index, 'groundtruth': groundtruth}


def inference(image):
    print('inference')
    print(image)
    # temp_image = Image.open(image).convert('L')  # translating a color image to black and white
    temp_image = Image.open(image)
    temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
    temp_image = np.asarray(temp_image) / 255.0
    temp_image = temp_image.reshape([-1, FLAGS.image_size, FLAGS.image_size, 3])
    tf.reset_default_graph()
    with tf.Session() as sess:
        logger.info('========start inference============')
        # images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
        # Pass a shadow label 0. This label will not affect the computation graph.

        graph = build_graph(top_k=5)
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print("restore from the checkpoint {0}".format(ckpt))
        predict_val, predict_index = sess.run([graph['predicted_val_top_k'], graph['predicted_index_top_k']],
                                              feed_dict={graph['images']: temp_image, graph['keep_prob']: 1.0})
    return predict_val, predict_index
    # exit()


def main(_):
    print(FLAGS.mode)
    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == 'valid':
        dct = validation()
        result_file = 'result.dict'
        logger.info('Write result into {0}'.format(result_file))
        with open(result_file, 'wb') as f:
            pickle.dump(dct, f)  # Write a pickled representation of obj to the open file object file.
        logger.info('Write file ends')
    elif FLAGS.mode == 'test':
        image_path = FLAGS.test_dir
        final_predict_val, final_predict_index = inference(image_path)
        logger.info('the result info label {0} predict index {1} predict_val {2}'.format(190, final_predict_index,
                                                                                         final_predict_val))


if __name__ == "__main__":
    tf.app.run()
