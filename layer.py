import numpy as np
import tensorflow as tf


def weight_variable(shape, mean=0.0, name=None):
    return tf.get_variable(
        name+"/weight",
        shape=shape,
        # initializer=tf.contrib.layers.xavier_initializer(),
        initializer=tf.truncated_normal_initializer(mean=mean, stddev=0.02),
        dtype=tf.float32
    )


def bias_variable(shape, name=None):
    return tf.get_variable(
        name+"/bias",
        shape=shape,
        initializer=tf.constant_initializer(0.0),
        dtype=tf.float32
    )


def conv2d(x, W, stride=None, padding=None):
    stride = stride or [1, 1, 1, 1]
    padding = padding or 'SAME'
    return tf.nn.conv2d(x, W, strides=stride, padding=padding)

def batch_normalization(x, is_training, scope='bn', epsilon=1e-5, decay=0.9):
    return tf.contrib.layers.batch_norm(
        x,
        decay=decay,
        updates_collections=None,
        epsilon=epsilon,
        scale=True,
        is_training=is_training,
        scope=scope
    )

