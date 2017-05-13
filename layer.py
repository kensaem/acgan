import numpy as np
import tensorflow as tf


def weight_variable(shape, name=None):
    return tf.get_variable(
        name+"/weight",
        shape=shape,
        initializer=tf.contrib.layers.xavier_initializer(),
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


def get_pop_variable(size, suffix=None):
    if suffix is None:
        suffix = ""
    else:
        suffix = "_" + suffix
    pop_mean = tf.get_variable('pop_mean'+suffix, [size], initializer=tf.zeros_initializer(tf.float32), trainable=False)
    pop_var = tf.get_variable('pop_var'+suffix, [size], initializer=tf.ones_initializer(tf.float32), trainable=False)
    return pop_mean, pop_var


def batch_normalization(x, is_training, scope='bn', pop_mean=None, pop_var=None, epsilon=1e-3, decay=0.99):
    with tf.variable_scope(scope):
        shape = x.get_shape()
        size = shape.as_list()[-1]
        axes = list(range(len(shape)-1))

        scale = tf.get_variable('scale', [size], initializer=tf.constant_initializer(0.1))
        offset = tf.get_variable('offset', [size])

        if pop_mean is None and pop_var is None:
            pop_mean = tf.get_variable('pop_mean', [size], initializer=tf.zeros_initializer(tf.float32), trainable=False)
            pop_var = tf.get_variable('pop_var', [size], initializer=tf.ones_initializer(tf.float32), trainable=False)

        batch_mean, batch_var = tf.nn.moments(x, axes)

        def batch_statistics():
            train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)

        def population_statistics():
            return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

        return tf.cond(is_training, batch_statistics, population_statistics)

