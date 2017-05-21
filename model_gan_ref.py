import tensorflow as tf
from layer import *


def leaky_relu(x, neg_thres=0.2):
    y = tf.maximum(x, neg_thres*x)
    return y


def cnn_block(
        input_t,
        output_channel,
        layer_name,
        kernel_size=4,
        stride=1,
        activation=leaky_relu,
        is_training=None,
        with_bn=True,
        padding='SAME',

):
    input_channel = input_t.get_shape().as_list()[-1]
    output_t = input_t
    with tf.variable_scope(layer_name):
        w_conv = weight_variable([kernel_size, kernel_size, input_channel, output_channel], name=layer_name)
        output_t = conv2d(output_t, w_conv, stride=[1, stride, stride, 1], padding=padding)
        if with_bn:
            output_t = batch_normalization(x=output_t, is_training=is_training, scope="bn")
        else:
            b_conv = bias_variable([output_channel], name=layer_name)
            output_t = output_t + b_conv
        output_t = activation(output_t)
    return output_t


def deconv_block(
        input_t,
        output_shape,
        layer_name,
        is_training,
        stride_size=2,
        kernel_size=4,
        activation=leaky_relu,
        with_bn=True,
        padding='SAME'
):
    inpus_channels = input_t.get_shape().as_list()[-1]
    with tf.variable_scope(layer_name):
        w_conv = weight_variable([kernel_size, kernel_size, output_shape[-1], inpus_channels], name=layer_name)
        output_t = tf.nn.conv2d_transpose(input_t, w_conv, output_shape, [1, stride_size, stride_size, 1], padding=padding)# + b_conv
        if with_bn:
            output_t = batch_normalization(output_t, is_training=is_training, scope='bn')
        else:
            b_conv = bias_variable([output_shape[-1]], name=layer_name)
            output_t = output_t + b_conv
        output_t = activation(output_t)
    return output_t


class ACGANModel:
    def __init__(self, batch_size=64):

        self.noise_size = 128
        self.cond_size = 5
        self.latent_cls_size = 10
        self.batch_size = batch_size

        self.global_step_disc = tf.Variable(0, trainable=False, name='global_step_discriminator')
        self.global_step_gen = tf.Variable(0, trainable=False, name='global_step_generator')
        self.keep_prob_ph = tf.placeholder(dtype=tf.float32, name='keep_probability')
        self.lr_gen_ph = tf.placeholder(dtype=tf.float32, name='learning_rate_for_generator')
        self.lr_disc_ph = tf.placeholder(dtype=tf.float32, name='learning_rate_for_discriminator')
        self.is_training_gen_ph = tf.placeholder(dtype=tf.bool, name='is_training_for_generator')
        self.is_training_disc_ph = tf.placeholder(dtype=tf.bool, name='is_training_for_discriminator')

        # FIXME temporary placeholder for convolutional dropout.
        self.batch_size_ph = tf.placeholder(
            dtype=tf.int32,
            shape=(),
            name='batch_size_placeholder'
        )

        self.input_image_ph = tf.placeholder(
            dtype=tf.uint8,
            shape=[None, 32, 32, 3],
            name='input_image_placeholder')

        self.label_cls_ph = tf.placeholder(
            dtype=tf.int64,
            shape=[None],
            name='label_class_placeholder')

        self.cond_ph = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.cond_size],
            name='label_condition_placeholder')

        # NOTE Build model for generator
        self.noise_t = tf.random_normal((self.batch_size_ph, self.noise_size))
        self.label_cond_t = tf.random_normal((self.batch_size_ph, self.cond_size))
        sample_cls = tf.multinomial(tf.ones((self.batch_size_ph, 10), dtype=tf.float32) / 10, 1)
        self.label_fake_cls_t = tf.to_int32(tf.squeeze(sample_cls, -1))

        self.fake_image_t = self.build_generator(
            self.label_fake_cls_t,
            self.noise_t,
            self.label_cond_t,
            batch_size=self.batch_size
        )
        print(self.fake_image_t)
        print(self.label_fake_cls_t)

        # NOTE Build model for discriminator
        self.real_image_t = tf.div(tf.to_float(self.input_image_ph), 127.5) - 1.0
        self.real_cls_t, self.real_disc_t, self.real_cond_t = self.build_discriminator(input_tensor=self.real_image_t)
        self.fake_cls_t, self.fake_disc_t, self.fake_cond_t = self.build_discriminator(input_tensor=self.fake_image_t, reuse=True)

        # NOTE Classification loss
        self.cls_loss_real = self.build_cls_loss(self.label_cls_ph, self.real_cls_t)
        self.cls_loss_fake = self.build_cls_loss(self.label_fake_cls_t, self.fake_cls_t)

        # NOTE Discriminator loss
        print(self.real_disc_t)
        noisy_real_labels = tf.random_uniform([self.batch_size_ph], minval=0.7, maxval=1.2)
        self.disc_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.real_disc_t,
                labels=noisy_real_labels,
                # labels=tf.ones_like(self.real_disc_t),
            )
        )
        noisy_fake_labels = tf.random_uniform([self.batch_size_ph], minval=0.0, maxval=0.3)
        self.disc_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.fake_disc_t,
                labels=noisy_fake_labels,
                # labels=tf.zeros_like(self.fake_disc_t),
            )
        )

        # NOTE Generator loss
        noisy_real_labels = tf.random_uniform([self.batch_size_ph], minval=0.7, maxval=1.2)
        self.gen_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.fake_disc_t,
                labels=noisy_real_labels,
                # labels=tf.ones_like(self.fake_disc_t),
            )
        )

        # condition loss
        self.cond_loss = tf.losses.mean_squared_error(
            self.label_cond_t,
            self.real_cond_t
        )


        # NOTE build optimizers
        beta1 = 0.5
        t_vars = tf.trainable_variables()
        # for var in t_vars:
        #     print(var.name)
        d_vars = [var for var in t_vars if "discriminator" in var.name]
        for var in d_vars:
            print(var.name)
        self.train_op_disc_real = tf.train.AdamOptimizer(self.lr_disc_ph, beta1=beta1).minimize(
            self.disc_loss_real,
            # (self.disc_loss_real + self.cls_loss_real),
            # (self.disc_loss_real + self.cls_loss_real + self.cond_loss),
            global_step=self.global_step_disc,
            var_list=d_vars,
        )

        self.train_op_disc_fake = tf.train.AdamOptimizer(self.lr_disc_ph, beta1=beta1).minimize(
            self.disc_loss_fake,
            # (self.disc_loss_fake + self.cls_loss_fake),
            # (self.disc_loss_fake + self.cls_loss_fake + self.cond_loss),
            global_step=self.global_step_disc,
            var_list=d_vars,
        )

        g_vars = [var for var in t_vars if "generator" in var.name]
        for var in g_vars:
            print(var.name)
        self.train_op_gen = tf.train.AdamOptimizer(self.lr_gen_ph, beta1=beta1).minimize(
            self.gen_loss,
            # (self.gen_loss + self.cls_loss_fake),
            # (self.gen_loss + self.cls_loss_fake + self.cond_loss),
            global_step=self.global_step_gen,
            var_list=g_vars,
        )

        # For validation and test
        most_realistic_index = tf.arg_max(self.fake_disc_t, 0)
        # most_realistic_index, most_realistic_disc_t = tf.nn.top_k(self.fake_disc_t, 1)
        self.most_realistic_fake_class = tf.gather(self.label_fake_cls_t, most_realistic_index)
        self.most_realistic_fake_image = tf.gather(self.fake_image_t, most_realistic_index)

        print(self.real_cls_t)
        self.pred_label = tf.arg_max(tf.nn.softmax(self.real_cls_t), 1)
        self.conf_matrix = tf.confusion_matrix(self.label_cls_ph, self.pred_label, num_classes=10)
        self.correct_count = tf.reduce_sum(tf.to_float(tf.equal(self.pred_label, self.label_cls_ph)), axis=0)
        print(self.pred_label)

        return

    def build_generator(
            self,
            input_class_t,
            input_noise_t,
            input_cond_t,
            batch_size,
            reuse=False,
            activation=tf.nn.relu,
            name="generator",
    ):
        # activation = leaky_relu

        with tf.variable_scope(name, reuse=reuse):
            with tf.variable_scope("embedding"):
                output_t = input_noise_t

                # class_embed_t = tf.one_hot(input_class_t, self.latent_cls_size)
                # output_t = tf.concat([input_noise_t, class_embed_t], axis=-1)

                # params = weight_variable([10, self.noise_size], name="class_params")
                # class_embed_t = tf.nn.embedding_lookup(params, input_class_t)
                # output_t = input_noise_t * class_embed_t

            output_t = tf.reshape(output_t, [-1, 1, 1, self.noise_size])
            output_t = deconv_block(
                output_t,
                [batch_size, 2, 2, 512],
                layer_name="conv_tp_0",
                is_training=self.is_training_gen_ph,
                activation=activation,
                stride_size=1,
                kernel_size=2,
                padding='VALID'
            )

            output_t = deconv_block(
                output_t,
                [batch_size, 4, 4, 256],
                layer_name="conv_tp_1",
                is_training=self.is_training_gen_ph,
                activation=activation,
            )

            output_t = deconv_block(
                output_t,
                [batch_size, 8, 8, 128],
                layer_name="conv_tp_2",
                is_training=self.is_training_gen_ph,
                activation=activation,
            )

            output_t = deconv_block(
                output_t,
                [batch_size, 16, 16, 64],
                layer_name="conv_tp_3",
                is_training=self.is_training_gen_ph,
                activation=activation,
            )

            output_t = deconv_block(
                output_t,
                [batch_size, 32, 32, 3],
                layer_name="conv_tp_4",
                is_training=self.is_training_gen_ph,
                with_bn=False,
                activation=tf.nn.tanh,
            )

        return output_t

    def build_discriminator(
            self,
            input_tensor,
            reuse=False,
            name="discriminator"
    ):
        activation = leaky_relu
        output_t = tf.cond(
            self.is_training_disc_ph,
            lambda: tf.map_fn(lambda img: tf.image.random_flip_left_right(img), input_tensor),
            lambda: input_tensor
        )

        with tf.variable_scope(name, reuse=reuse):
            # input size 32
            output_t = cnn_block(
                input_t=output_t,
                output_channel=64,
                layer_name="layer1",
                is_training=self.is_training_disc_ph,
                activation=activation,
                stride=2,
            )

            # input size 16
            output_t = cnn_block(
                input_t=output_t,
                output_channel=128,
                layer_name="layer2",
                is_training=self.is_training_disc_ph,
                activation=activation,
                stride=2,
            )

            # input size 8
            output_t = cnn_block(
                input_t=output_t,
                output_channel=256,
                layer_name="layer3",
                is_training=self.is_training_disc_ph,
                activation=activation,
                stride=2,
            )

            # input size 4
            # output_t = cnn_block(
            #     input_tensor=output_t,
            #     output_channel=512,
            #     layer_name="layer4",
            #     is_training=self.is_training_disc_ph,
            #     activation=activation,
            #     stride=2,
            # )

            # input size 2 w/ channel 512 => fc layer
            output_share_t = output_t

            # Branch 1. classification for label [0~9]
            output_cls_t = output_share_t
            output_cls_t = tf.reshape(output_cls_t, [-1, 256*4*4])
            with tf.variable_scope("cls_fc_1"):
                w_fc = weight_variable([256*4*4, 512], name="cls_fc_1")
                b_fc = bias_variable([512], name="cls_fc_1")
                output_cls_t = tf.matmul(output_cls_t, w_fc) + b_fc
                output_cls_t = batch_normalization(x=output_cls_t, is_training=self.is_training_disc_ph, scope="bn")
                output_cls_t = activation(output_cls_t)

            with tf.variable_scope("cls_fc_2"):
                w_fc = weight_variable([512, 10], name="cls_fc_2")
                b_fc = bias_variable([10], name="cls_fc_2")
                output_cls_t = tf.matmul(output_cls_t, w_fc) + b_fc

            # Branch 2. discriminator for real / fake
            output_disc_t = output_share_t
            with tf.variable_scope("disc_conv"):
                w_conv = weight_variable([4, 4, 256, 1], name="disc_conv")
                b_conv = bias_variable([1], name="disc_conv")
                output_disc_t = conv2d(output_disc_t, w_conv, padding='VALID') + b_conv

                print(output_disc_t)
                output_disc_t = tf.reshape(output_disc_t, [-1])
                print(output_disc_t)

            # Branch 3. latent
            output_cond_t = output_share_t
            output_cond_t = tf.reshape(output_cond_t, [-1, 256*4*4])
            with tf.variable_scope("cond_fc"):
                w_fc = weight_variable([256*4*4, self.cond_size], name="cond_fc")
                b_fc = bias_variable([self.cond_size], name="cond_fc")
                output_cond_t = tf.matmul(output_cond_t, w_fc) + b_fc

        return output_cls_t, output_disc_t, output_cond_t

    def build_cls_loss(self, labels, outputs):

        # each_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=outputs)
        # accum_loss = tf.reduce_mean(each_loss, axis=[0])

        # make soft soft-max cross entropy
        onehot_labels = tf.one_hot(labels, 10)
        accum_loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=outputs, label_smoothing=0)

        return accum_loss

