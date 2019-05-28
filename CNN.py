import logging
import os
import pickle
import random
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image


def cnn(images):

    conv_1 = slim.conv2d(images, 64, [3, 3], 1, padding='SAME', scope='conv1')
    max_pool_1 = slim.max_pool2d(conv_1, [2, 2], [2, 2], padding='SAME')
    conv_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding='SAME', scope='conv2')
    max_pool_2 = slim.max_pool2d(conv_2, [2, 2], [2, 2], padding='SAME')
    conv_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding='SAME', scope='conv3')
    max_pool_3 = slim.max_pool2d(conv_3, [2, 2], [2, 2], padding='SAME')

    flatten = slim.flatten(max_pool_3)
    fc1 = slim.fully_connected(slim.dropout(flatten, 0.5), 1024, activation_fn=tf.nn.relu, scope='fc1')

    return fc1
