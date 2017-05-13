import tensorflow as tf
from layer import *
from vgg_model import *


def leaky_relu(x):
    return tf.where(tf.greater(x, 0), x, 0.01 * x)


class ACGANModel:
    def __init__(self):

        self.noise_size = 100
        self.cond_size = 5
        self.latent_cls_size = 10

        self.global_step_disc = tf.Variable(0, trainable=False, name='global_step_discriminator')
        self.global_step_gen = tf.Variable(0, trainable=False, name='global_step_generator')
        self.keep_prob_ph = tf.placeholder(dtype=tf.float32, name='keep_probability')
        self.lr_gen_ph = tf.placeholder(dtype=tf.float32, name='learning_rate_for_generator')
        self.lr_disc_ph = tf.placeholder(dtype=tf.float32, name='learning_rate_for_discriminator')
        self.is_training_ph = tf.placeholder(dtype=tf.bool, name='is_training')

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

        self.noise_t = tf.random_normal((self.batch_size_ph, self.noise_size))

        # Build model for generator
        sample_cls = tf.multinomial(tf.ones((self.batch_size_ph, 10), dtype=tf.float32) / 10, 1)
        self.label_fake_cls_t = tf.to_int32(tf.squeeze(sample_cls, -1))
        self.fake_image_t = self.build_generator(
            self.label_fake_cls_t,
            self.noise_t,
            batch_size=50
        )
        print(self.fake_image_t)

        # Build model for discriminator
        self.fake_cls_t, self.fake_disc_t, self.fake_cond_t = self.build_discriminator(
            input_tensor=self.fake_image_t,
        )

        input_real_tensor = tf.div(tf.to_float(self.input_image_ph), 255.0, name="input_image_float")
        self.real_cls_t, self.real_disc_t, _ = self.build_discriminator(
            input_tensor=input_real_tensor,
            reuse=True,
        )

        # Classification loss
        self.real_cls_loss = self.build_cls_loss(self.label_cls_ph, self.real_cls_t)
        self.fake_cls_loss = self.build_cls_loss(self.label_fake_cls_t, self.fake_cls_t)
        self.cls_loss = (self.real_cls_loss + self.fake_cls_loss) / 2.0

        # Discrimination loss
        self.real_disc_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.real_disc_t,
                labels=tf.ones_like(self.real_disc_t, name="real")
            )
        )
        self.fake_disc_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.fake_disc_t,
                labels=tf.zeros_like(self.fake_disc_t)
            )
        )
        self.disc_loss = (self.real_disc_loss * self.fake_disc_loss) / 2.0

        # Extra discrimination loss for generator
        self.disc_for_gen_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.fake_disc_t,
                labels=tf.ones_like(self.fake_disc_t)
            )
        )

        # Latent loss
        self.cond_loss = tf.losses.mean_squared_error(
            self.noise_t,
            self.fake_cond_t
        )

        self.train_op_disc = tf.train.GradientDescentOptimizer(self.lr_disc_ph).minimize(
            (self.disc_loss + self.cls_loss), # + self.cond_loss),
            global_step=self.global_step_disc,
        )

        self.train_op_gen = tf.train.AdamOptimizer(self.lr_gen_ph).minimize(
            (self.disc_for_gen_loss + self.cls_loss), # + self.cond_loss),
            global_step=self.global_step_gen,
        )


        # For validation and test
        print(self.real_cls_t)
        self.pred_label = tf.arg_max(tf.nn.softmax(self.real_cls_t), 1)
        self.conf_matrix = tf.confusion_matrix(self.label_cls_ph, self.pred_label, num_classes=10)
        self.correct_count = tf.reduce_sum(tf.to_float(tf.equal(self.pred_label, self.label_cls_ph)), axis=0)
        print(self.pred_label)

        return

    def build_generator(
            self,
            input_class_t,
            input_latent_t,
            batch_size,
            reuse=False,
            name="generator",
    ):
        activation = tf.nn.relu

        with tf.variable_scope(name, reuse=reuse):
            with tf.variable_scope("embedding"):
                params = weight_variable([10, self.latent_cls_size], name="class_params")
                class_embed_t = tf.nn.embedding_lookup(params, input_class_t)

            output_t = tf.concat([input_latent_t, class_embed_t], axis=-1)

            with tf.variable_scope("fc_1"):
                w_fc = weight_variable([self.noise_size+self.latent_cls_size, 1024], name="fc_1")
                output_t = tf.matmul(output_t, w_fc)
                output_t = batch_normalization(output_t, is_training=self.is_training_ph, scope='bn')
                output_t = activation(output_t)

            with tf.variable_scope("fc_2"):
                w_fc = weight_variable([1024, 4*4*128], name="fc_2")
                output_t = tf.matmul(output_t, w_fc)
                output_t = batch_normalization(output_t, is_training=self.is_training_ph, scope='bn')
                output_t = activation(output_t)

            output_t = tf.reshape(output_t, [-1, 4, 4, 128])

            with tf.variable_scope("conv_tp_1"):
                w_conv = weight_variable([5, 5, 64, 128], name="conv_tp_1")
                output_t = tf.nn.conv2d_transpose(output_t, w_conv, [batch_size, 8, 8, 64], [1, 2, 2, 1])
                output_t = batch_normalization(output_t, is_training=self.is_training_ph, scope='bn')
                output_t = activation(output_t)

            with tf.variable_scope("conv_tp_2"):
                w_conv = weight_variable([5, 5, 16, 64], name="conv_tp_2")
                output_t = tf.nn.conv2d_transpose(output_t, w_conv, [batch_size, 16, 16, 16], [1, 2, 2, 1])
                output_t = batch_normalization(output_t, is_training=self.is_training_ph, scope='bn')
                output_t = activation(output_t)

            with tf.variable_scope("conv_tp_3"):
                w_conv = weight_variable([5, 5, 3, 16], name="conv_tp_3")
                b_conv = bias_variable([3], name="conv_tp_3")
                output_t = tf.nn.conv2d_transpose(output_t, w_conv, [batch_size, 32, 32, 3], [1, 2, 2, 1])
                output_t += b_conv

            # last activation is sigmoid
            output_t = tf.nn.sigmoid(output_t)

        return output_t

    def build_discriminator(
            self,
            input_tensor,
            reuse=False,
            name="discriminator"
    ):
        layers_size = [1, 1, 2, 2, 2]  #vgg11
        # layers_size = [2, 2, 3, 3, 3]  #vgg16
        # layers_size = [2, 2, 4, 4, 4]  #vgg19

        output_tensor = tf.cond(
            self.is_training_ph,
            lambda: tf.map_fn(lambda img: tf.image.random_flip_left_right(img), input_tensor),
            lambda: input_tensor
        )

        with tf.variable_scope(name, reuse=reuse):
            # input size 32
            for idx in range(layers_size[0]):
                output_tensor = vgg_block(
                    input_tensor=output_tensor,
                    output_channel=64,
                    layer_name="layer1_"+str(idx),
                    is_training=self.is_training_ph,
                    activation=leaky_relu,
                )
            with tf.variable_scope("layer1_3"):
                output_tensor = tf.nn.max_pool(output_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', data_format='NHWC')

            output_tensor = tf.nn.dropout(
                output_tensor,
                self.keep_prob_ph,
                noise_shape=[self.batch_size_ph, 1, 1, 64]
            )

            # input size 16
            for idx in range(layers_size[1]):
                output_tensor = vgg_block(
                    input_tensor=output_tensor,
                    output_channel=128,
                    layer_name="layer2_"+str(idx),
                    is_training=self.is_training_ph,
                    activation=leaky_relu,
                )
            with tf.variable_scope("layer2_pooling"):
                output_tensor = tf.nn.max_pool(output_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', data_format='NHWC')

            output_tensor = tf.nn.dropout(
                output_tensor,
                self.keep_prob_ph,
                noise_shape=[self.batch_size_ph, 1, 1, 128]
            )

            # input size 8
            for idx in range(layers_size[2]):
                output_tensor = vgg_block(
                    input_tensor=output_tensor,
                    output_channel=256,
                    layer_name="layer3_"+str(idx),
                    is_training=self.is_training_ph,
                    activation=leaky_relu,
                )
            with tf.variable_scope("layer3_pooling"):
                output_tensor = tf.nn.max_pool(output_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', data_format='NHWC')

            output_tensor = tf.nn.dropout(
                output_tensor,
                self.keep_prob_ph,
                noise_shape=[self.batch_size_ph, 1, 1, 256]
            )

            # input size 4
            for idx in range(layers_size[3]):
                output_tensor = vgg_block(
                    input_tensor=output_tensor,
                    output_channel=512,
                    layer_name="layer4_"+str(idx),
                    is_training=self.is_training_ph,
                )
            with tf.variable_scope("layer4_pooling"):
                output_tensor = tf.nn.max_pool(output_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', data_format='NHWC')

            output_tensor = tf.nn.dropout(
                output_tensor,
                self.keep_prob_ph,
                noise_shape=[self.batch_size_ph, 1, 1, 512]
            )

            # input size 2
            for idx in range(layers_size[4]):
                output_tensor = vgg_block(
                    input_tensor=output_tensor,
                    output_channel=512,
                    layer_name="layer5_"+str(idx),
                    is_training=self.is_training_ph,
                )
            with tf.variable_scope("layer5_pooling"):
                output_tensor = tf.nn.max_pool(output_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', data_format='NHWC')

            output_tensor = tf.nn.dropout(
                output_tensor,
                self.keep_prob_ph,
                noise_shape=[self.batch_size_ph, 1, 1, 512]
            )

            # input size 1 w/ channel 512 => fc layer
            output_share_t = tf.squeeze(output_tensor, axis=[1, 2])

            # Branch 1. classification for label [0~9]
            output_class_t = output_share_t
            with tf.variable_scope("classification_fc_1"):
                w_fc = weight_variable([512, 512], name="classification_fc_1")
                output_class_t = tf.matmul(output_class_t, w_fc)
                output_class_t = batch_normalization(x=output_class_t, is_training=self.is_training_ph, scope="bn")
                output_class_t = leaky_relu(output_class_t)

            output_class_t = tf.nn.dropout(output_class_t, self.keep_prob_ph)

            with tf.variable_scope("classification_fc_2"):
                w_fc = weight_variable([512, 10], name="classification_fc_2")
                b_fc = bias_variable([10], name="classification_fc_2")
                output_class_t = tf.matmul(output_class_t, w_fc) + b_fc

            # Branch 2. discriminator for real / fake
            output_discriminator_t = output_share_t
            with tf.variable_scope("discriminator_fc_1"):
                w_fc = weight_variable([512, 10], name="discriminator_fc_1")
                b_fc = bias_variable([10], name="discriminator_fc_1")
                output_discriminator_t = tf.matmul(output_discriminator_t, w_fc) + b_fc
                output_discriminator_t = batch_normalization(x=output_discriminator_t, is_training=self.is_training_ph, scope="bn")
                output_discriminator_t = leaky_relu(output_discriminator_t)

            with tf.variable_scope("discriminator_fc_2"):
                w_fc = weight_variable([10, 1], name="discriminator_fc_2")
                b_fc = bias_variable([1], name="discriminator_fc_2")
                output_discriminator_t = tf.matmul(output_discriminator_t, w_fc) + b_fc

            # Branch 3. latent
            output_latent_t = output_share_t
            with tf.variable_scope("latent_fc"):
                w_fc = weight_variable([512, self.noise_size], name="latent_fc")
                b_fc = bias_variable([self.noise_size], name="latent_fc")
                output_latent_t = tf.matmul(output_latent_t, w_fc) + b_fc

        return output_class_t, output_discriminator_t, output_latent_t

    def build_cls_loss(self, labels, outputs):
        each_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=outputs)
        accum_loss = tf.reduce_mean(each_loss, axis=[0])
        return accum_loss


