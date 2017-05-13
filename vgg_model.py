import tensorflow as tf
from layer import *


def vgg_block(
        input_tensor,
        output_channel,
        layer_name,
        activation=tf.nn.relu,
        is_training=None,
        with_bn=True,
        with_dropout=False,
        keep_prob_placeholder=None,
):
    if with_bn:
        assert(is_training is not None)

    input_channel = input_tensor.get_shape().as_list()[-1]
    batch_size = tf.shape(input_tensor)[0]

    output_tensor = input_tensor
    with tf.variable_scope(layer_name):
        w_conv = weight_variable([3, 3, input_channel, output_channel], name=layer_name)
        output_tensor = conv2d(output_tensor, w_conv, stride=[1, 1, 1, 1], padding='SAME')
        if with_bn:
            output_tensor = batch_normalization(x=output_tensor, is_training=is_training, scope="bn")
        else:
            b_conv = bias_variable([output_channel], name=layer_name)
            output_tensor += b_conv
        output_tensor = activation(output_tensor)
        if with_dropout \
                and keep_prob_placeholder is not None \
                and batch_size is not None:
            output_tensor = tf.nn.dropout(
                output_tensor,
                keep_prob_placeholder,
                noise_shape=[batch_size, 1, 1, output_channel]
            )
    return output_tensor


class Model:
    def __init__(self, optim, with_bn, with_dropout, with_residual=False):
        self.optim = optim

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.keep_prob_placeholder = tf.placeholder(dtype=tf.float32, name='keep_probability')
        self.lr_placeholder = tf.placeholder(dtype=tf.float32, name='learning_rate')
        self.is_training_placeholder = tf.placeholder(dtype=tf.bool, name='is_training')
        self.input_image_placeholder = tf.placeholder(
            dtype=tf.uint8,
            shape=[None, 32, 32, 3],
            name='input_image_placeholder')

        self.label_placeholder = tf.placeholder(
            dtype=tf.int64,
            shape=[None],
            name='label_placeholder')

        # FIXME temporary placeholder for convolutional dropout.
        self.batch_size_placeholder = tf.placeholder(
            dtype=tf.int32,
            shape=(),
            name='batch_size_placeholder'
        )

        self.output = self.build_model_vgg(
            with_bn=with_bn,
            with_dropout=with_dropout,
        )
        self.each_loss, self.accum_loss = self.build_loss()
        self.pred_label = tf.arg_max(tf.nn.softmax(self.output), 1)

        if self.optim == "sgd":
            optimizer = tf.train.GradientDescentOptimizer
        elif self.optim == "adam":
            optimizer = tf.train.AdamOptimizer
        else:
            assert("Unknown optimizer")
        self.train_op = optimizer(self.lr_placeholder).minimize(
            self.accum_loss,
            global_step=self.global_step,
        )

        self.conf_matrix = tf.confusion_matrix(self.label_placeholder, self.pred_label, num_classes=10)
        self.correct_count = tf.reduce_sum(tf.to_float(tf.equal(self.pred_label, self.label_placeholder)), axis=0)
        print(self.each_loss, self.accum_loss)
        print(self.pred_label)

        return

    def build_model_vgg(self, with_bn, with_dropout, name="vgg"):
        # layers_size = [1, 1, 2, 2, 2]  #vgg11
        layers_size = [2, 2, 3, 3, 3]  #vgg16
        # layers_size = [2, 2, 4, 4, 4]  #vgg19

        output_tensor = self.input_image_placeholder
        output_tensor = tf.cond(
            self.is_training_placeholder,
            lambda: tf.map_fn(lambda img: tf.image.random_flip_left_right(img), output_tensor),
            lambda: output_tensor
        )
        output_tensor = tf.div(tf.to_float(output_tensor), 255.0, name="input_image_float")
        print(output_tensor)

        with tf.variable_scope(name):

            # input size 32
            for idx in range(layers_size[0]):
                output_tensor = vgg_block(
                    input_tensor=output_tensor,
                    output_channel=64,
                    layer_name="layer1_"+str(idx),
                    is_training=self.is_training_placeholder,
                    with_bn=with_bn,
                )
            with tf.variable_scope("layer1_3"):
                output_tensor = tf.nn.max_pool(output_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', data_format='NHWC')

            if with_dropout:
                output_tensor = tf.nn.dropout(
                    output_tensor,
                    self.keep_prob_placeholder,
                    noise_shape=[self.batch_size_placeholder, 1, 1, 64]
                )

            # input size 16
            for idx in range(layers_size[1]):
                output_tensor = vgg_block(
                    input_tensor=output_tensor,
                    output_channel=128,
                    layer_name="layer2_"+str(idx),
                    is_training=self.is_training_placeholder,
                    with_bn=with_bn,
                )
            with tf.variable_scope("layer2_pooling"):
                output_tensor = tf.nn.max_pool(output_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', data_format='NHWC')

            if with_dropout:
                output_tensor = tf.nn.dropout(
                    output_tensor,
                    self.keep_prob_placeholder,
                    noise_shape=[self.batch_size_placeholder, 1, 1, 128]
                )

            # input size 8
            for idx in range(layers_size[2]):
                output_tensor = vgg_block(
                    input_tensor=output_tensor,
                    output_channel=256,
                    layer_name="layer3_"+str(idx),
                    is_training=self.is_training_placeholder,
                    with_bn=with_bn,
                )
            with tf.variable_scope("layer3_pooling"):
                output_tensor = tf.nn.max_pool(output_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', data_format='NHWC')

            if with_dropout:
                output_tensor = tf.nn.dropout(
                    output_tensor,
                    self.keep_prob_placeholder,
                    noise_shape=[self.batch_size_placeholder, 1, 1, 256]
                )

            # input size 4
            for idx in range(layers_size[3]):
                output_tensor = vgg_block(
                    input_tensor=output_tensor,
                    output_channel=512,
                    layer_name="layer4_"+str(idx),
                    is_training=self.is_training_placeholder,
                    with_bn=with_bn,
                )
            with tf.variable_scope("layer4_pooling"):
                output_tensor = tf.nn.max_pool(output_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', data_format='NHWC')

            if with_dropout:
                output_tensor = tf.nn.dropout(
                    output_tensor,
                    self.keep_prob_placeholder,
                    noise_shape=[self.batch_size_placeholder, 1, 1, 512]
                )

            # input size 2
            for idx in range(layers_size[4]):
                output_tensor = vgg_block(
                    input_tensor=output_tensor,
                    output_channel=512,
                    layer_name="layer5_"+str(idx),
                    is_training=self.is_training_placeholder,
                    with_bn=with_bn,
                )
            with tf.variable_scope("layer5_pooling"):
                output_tensor = tf.nn.max_pool(output_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', data_format='NHWC')

            if with_dropout:
                output_tensor = tf.nn.dropout(
                    output_tensor,
                    self.keep_prob_placeholder,
                    noise_shape=[self.batch_size_placeholder, 1, 1, 512]
                )

            # input size 1 w/ channel 512 => fc layer
            output_tensor = tf.squeeze(output_tensor, axis=[1, 2])

            with tf.variable_scope("fc_1"):
                w_fc1 = weight_variable([512, 512], name="fc_1")
                output_tensor = tf.matmul(output_tensor, w_fc1)
                if with_bn:
                    output_tensor = batch_normalization(x=output_tensor, is_training=self.is_training_placeholder, scope="bn")
                else:
                    b_fc1 = bias_variable([512], name="fc_1")
                    output_tensor += b_fc1
                output_tensor = tf.nn.relu(output_tensor)

            if with_dropout:
                output_tensor = tf.nn.dropout(output_tensor, self.keep_prob_placeholder)

            with tf.variable_scope("fc_2"):

                w_fc2 = weight_variable([512, 10], name="fc_2")
                b_fc2 = bias_variable([10], name="fc_2")
                output_tensor = tf.matmul(output_tensor, w_fc2) + b_fc2

            print("last layer of VGG16 with BN and DROPOUT =", output_tensor)

        return output_tensor

    def build_loss(self):
        each_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_placeholder, logits=self.output)
        accum_loss = tf.reduce_mean(each_loss, axis=[0])
        return each_loss, accum_loss


