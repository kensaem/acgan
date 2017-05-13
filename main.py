import os
import shutil
import sys
import time
import tensorflow as tf

from loader import *
from loader_upsampling import *
from vgg_model import *
from gan_model import *
import cv2

tf.app.flags.DEFINE_string('data_path', '../data', 'Directory path to read the data files')
tf.app.flags.DEFINE_string('checkpoint_path', 'model', 'Directory path to save checkpoint files')

tf.app.flags.DEFINE_boolean('train_continue', False, 'flag for continue training from previous checkpoint')
tf.app.flags.DEFINE_boolean('valid_only', False, 'flag for validation only. this will make train_continue flag ignored')

tf.app.flags.DEFINE_integer('batch_size', 50, 'mini-batch size for training')
tf.app.flags.DEFINE_float('lr_gen', 1e-3, 'initial learning rate for generator')   # 1e-3 for adam
tf.app.flags.DEFINE_float('lr_disc', 1e-1, 'initial learning rate for discriminator')   # 1e-3 for adam
tf.app.flags.DEFINE_float('lr_decay_ratio', 0.95, 'ratio for decaying learning rate')
tf.app.flags.DEFINE_integer('lr_decay_interval', 2000, 'step interval for decaying learning rate')
tf.app.flags.DEFINE_integer('train_log_interval', 100, 'step interval for triggering print logs of train')
tf.app.flags.DEFINE_integer('valid_log_interval', 500, 'step interval for triggering validation')

tf.app.flags.DEFINE_boolean('use_bn', True, 'use batch normalization or not')
tf.app.flags.DEFINE_boolean('use_dropout', True, 'use drop-out or not')
tf.app.flags.DEFINE_boolean('use_ohem', False, 'use OHEM or not')
tf.app.flags.DEFINE_integer('hard_sampling_factor', 2, 'searching window size for OHEM')

tf.app.flags.DEFINE_boolean('use_upsampling', False, 'use upsampling for loader or not')

FLAGS = tf.app.flags.FLAGS


class Classifier:
    def __init__(self):
        self.sess = tf.Session()
        # self.model = Model(FLAGS.optim, FLAGS.use_bn, FLAGS.use_dropout)
        self.model = ACGANModel()
        self.batch_size = FLAGS.batch_size
        if FLAGS.use_upsampling:
            self.train_loader = LoaderUpsampling(data_path=os.path.join(FLAGS.data_path, "train"), batch_size=self.batch_size)
        else:
            self.train_loader = Loader(data_path=os.path.join(FLAGS.data_path, "train"), batch_size=self.batch_size)
        self.valid_loader = Loader(data_path=os.path.join(FLAGS.data_path, "val"), batch_size=self.batch_size)

        self.epoch_counter = 1
        self.lr_gen = FLAGS.lr_gen
        self.lr_disc = FLAGS.lr_disc
        self.lr_decay_interval = FLAGS.lr_decay_interval
        self.lr_decay_ratio = FLAGS.lr_decay_ratio
        self.train_log_interval = FLAGS.train_log_interval
        self.valid_log_interval = FLAGS.valid_log_interval

        self.train_continue = FLAGS.train_continue or FLAGS.valid_only
        self.checkpoint_dirpath = FLAGS.checkpoint_path
        self.checkpoint_filepath = os.path.join(self.checkpoint_dirpath, 'model.ckpt')
        self.log_dirpath = "log"

        if not self.train_continue and os.path.exists(self.checkpoint_dirpath):
            shutil.rmtree(self.log_dirpath, ignore_errors=True)
            shutil.rmtree(self.checkpoint_dirpath, ignore_errors=True)
        if not os.path.exists(self.checkpoint_dirpath):
            os.makedirs(self.checkpoint_dirpath)

        self.train_summary_writer = tf.summary.FileWriter(
            os.path.join(self.log_dirpath, 'train'),
            self.sess.graph,
        )

        self.valid_summary_writer = tf.summary.FileWriter(
            os.path.join(self.log_dirpath, 'valid'),
            self.sess.graph
        )

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=3)

        if self.train_continue:
            print("======== Restoring from saved checkpoint ========")
            save_path = self.checkpoint_dirpath
            ckpt = tf.train.get_checkpoint_state(save_path)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                print("======>" + ckpt.model_checkpoint_path)
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)

                # Reset log of steps after model is saved.
                last_step = self.model.global_step_disc.eval(self.sess)
                session_log = tf.SessionLog(status=tf.SessionLog.START)
                self.train_summary_writer.add_session_log(session_log, last_step+1)
                self.valid_summary_writer.add_session_log(session_log, last_step+1)
        return

    def make_report(self, report_file_path, conf_matrix, disc_loss, gen_loss, accuracy):
        with open(report_file_path, 'w') as file:
            str_row = ""
            for col_idx in range(10):
                str_row += "\tpred_"+str(col_idx)
            str_row += "\n"
            file.write(str_row)
            for row_idx, row in enumerate(conf_matrix):
                str_row = "label_"+str(row_idx)
                for col_idx, col in enumerate(row):
                    str_row += "\t%4d"%col
                str_row += "\n"
                file.write(str_row)

            file.write("\n")
            file.write("discriminator loss = %f\n" % disc_loss)
            file.write("generator loss = %f\n" % gen_loss)
            file.write("accuracy = %f\n" % accuracy)
        return

    def train(self):
        self.train_loader.reset()

        accum_disc_loss = .0
        accum_gen_loss = .0
        accum_correct_count = .0
        accum_conf_matrix = None

        last_valid_disc_loss = sys.float_info.max
        last_valid_gen_loss = sys.float_info.max
        last_valid_accuracy = .0
        while True:
            start_time = time.time()
            batch_data = self.train_loader.get_batch()
            if batch_data is None:
                print('%d epoch complete' % self.epoch_counter)
                self.epoch_counter += 1
                continue

            # Step 1. train for discriminator
            sess_input = [
                self.model.train_op_disc,
                self.model.disc_loss,
                self.model.correct_count,
                self.model.conf_matrix,
                self.model.global_step_disc,
            ]
            sess_output = self.sess.run(
                fetches=sess_input,
                feed_dict={
                    self.model.lr_disc_ph: self.lr_disc,
                    self.model.keep_prob_ph: 0.7,
                    self.model.is_training_ph: True,
                    self.model.input_image_ph: batch_data.images,
                    self.model.label_cls_ph: batch_data.labels,
                    self.model.batch_size_ph: self.batch_size,
                }
            )

            cur_step = sess_output[-1]
            accum_disc_loss += sess_output[1]
            accum_correct_count += sess_output[2]
            if accum_conf_matrix is None:
                accum_conf_matrix = sess_output[3]
            else:
                accum_conf_matrix += sess_output[3]

            # Step 2. train for generator
            sess_input = [
                self.model.train_op_gen,
                self.model.disc_for_gen_loss,
                self.model.global_step_gen,
            ]
            sess_output = self.sess.run(
                fetches=sess_input,
                feed_dict={
                    self.model.lr_gen_ph: self.lr_gen,
                    self.model.keep_prob_ph: 0.7,
                    self.model.is_training_ph: True,
                    self.model.input_image_ph: batch_data.images,
                    self.model.label_cls_ph: batch_data.labels,
                    self.model.batch_size_ph: self.batch_size,
                }
            )
            accum_gen_loss += sess_output[1]


            if cur_step > 0 and cur_step % self.train_log_interval == 0:
                duration = time.time() - start_time
                disc_loss = accum_disc_loss / self.train_log_interval
                gen_loss = accum_gen_loss / self.train_log_interval
                accuracy = accum_correct_count / (self.batch_size * self.train_log_interval)

                print("[step %d] training : disc. loss = %f, gen. loss = %f, accuracy = %.6f (%.4f sec)"
                      % (cur_step, disc_loss, gen_loss, accuracy, duration))

                # log for tensorboard
                custom_summaries = [
                    tf.Summary.Value(tag='loss_disc', simple_value=disc_loss),
                    tf.Summary.Value(tag='loss_gen', simple_value=gen_loss),
                    tf.Summary.Value(tag='accuracy', simple_value=accuracy),
                    tf.Summary.Value(tag='learning rate for discriminator', simple_value=self.lr_disc),
                    tf.Summary.Value(tag='learning rate for generator', simple_value=self.lr_gen),
                ]
                self.train_summary_writer.add_summary(tf.Summary(value=custom_summaries), cur_step)
                self.train_summary_writer.flush()

                # reset local accumulations
                accum_disc_loss = .0
                accum_gen_loss = .0
                accum_correct_count = .0
                accum_conf_matrix = None

            if cur_step > 0 and cur_step % self.valid_log_interval == 0:
                cur_valid_disc_loss, cur_valid_gen_loss, cur_valid_accuracy, valid_conf_matrix = self.valid()
                print("... validation : disc. loss = %f, gen. loss = %f, accuracy = %.6f"
                      % (cur_valid_disc_loss, cur_valid_gen_loss, cur_valid_accuracy))
                print("==== confusion matrix ====")
                print(valid_conf_matrix)

                if cur_valid_disc_loss < last_valid_disc_loss and cur_valid_gen_loss < last_valid_gen_loss:
                    # Save confusion matrix to disk
                    report_file_path = os.path.join(self.log_dirpath, "report_%d" % cur_step+".txt")
                    self.make_report(
                        report_file_path,
                        valid_conf_matrix,
                        cur_valid_disc_loss,
                        cur_valid_gen_loss,
                        cur_valid_accuracy
                    )

                    # Save the variables to disk.
                    save_path = self.saver.save(
                        self.sess,
                        self.checkpoint_filepath+"_dl_%.4f"%cur_valid_disc_loss+"_gl_%.4f"%cur_valid_gen_loss+"_accuracy_%.4f"%cur_valid_accuracy,
                        global_step=cur_step
                    )
                    print("Model saved in file: %s" % save_path)
                    last_valid_disc_loss = cur_valid_disc_loss
                    last_valid_gen_loss = cur_valid_gen_loss
                    last_valid_accuracy = cur_valid_accuracy

            if cur_step > 0 and cur_step % self.lr_decay_interval == 0:
                self.lr_disc *= self.lr_decay_ratio
                self.lr_gen *= self.lr_decay_ratio
                print("\t===> learning rate decayed to %f, %f" % (self.lr_disc, self.lr_gen))

        return

    def valid(self):
        self.valid_loader.reset()

        step_counter = .0
        accum_disc_loss = .0
        accum_gen_loss = .0
        accum_correct_count = .0
        accum_conf_matrix = None

        fake_image = None
        fake_label = None

        valid_batch_size = self.batch_size
        while True:
            batch_data = self.valid_loader.get_batch(valid_batch_size)
            if batch_data is None:
                # print('%d validation complete' % self.epoch_counter)
                break

            sess_input = [
                self.model.disc_loss,
                self.model.disc_for_gen_loss,
                self.model.correct_count,
                self.model.conf_matrix,
                self.model.fake_image_t,
                self.model.fake_cls_t,
            ]
            sess_output = self.sess.run(
                fetches=sess_input,
                feed_dict={
                    self.model.keep_prob_ph: 1.0,
                    self.model.is_training_ph: False,
                    self.model.input_image_ph: batch_data.images,
                    self.model.label_cls_ph: batch_data.labels,
                    self.model.batch_size_ph: valid_batch_size,
                }
            )

            accum_disc_loss += sess_output[0]
            accum_gen_loss += sess_output[1]
            accum_correct_count += sess_output[2]
            if accum_conf_matrix is None:
                accum_conf_matrix = sess_output[3]
            else:
                accum_conf_matrix += sess_output[3]

            fake_image = sess_output[4][0] * 255.0
            fake_label = sess_output[5][0]

            step_counter += 1

        print("Class for fake image = %d" % fake_label)
        cv2.imwrite("fake_image.jpg", fake_image)

        disc_loss = accum_disc_loss / step_counter
        gen_loss = accum_gen_loss / step_counter
        accuracy = accum_correct_count / (step_counter * valid_batch_size)

        # log for tensorboard
        cur_step = self.sess.run(self.model.global_step_disc)
        custom_summaries = [
            tf.Summary.Value(tag='loss_disc', simple_value=disc_loss),
            tf.Summary.Value(tag='loss_gen', simple_value=gen_loss),
            tf.Summary.Value(tag='accuracy', simple_value=accuracy),
        ]
        self.valid_summary_writer.add_summary(tf.Summary(value=custom_summaries), cur_step)
        self.valid_summary_writer.flush()

        return disc_loss, gen_loss, accuracy, accum_conf_matrix


class ClassifierOHEM(Classifier):

    def __init__(self, hard_sampling_factor):
        super().__init__()
        self.hard_sampling_factor = hard_sampling_factor

    def train(self):
        self.train_loader.reset()

        accum_loss = .0
        accum_correct_count = .0
        accum_conf_matrix = None

        last_valid_loss = sys.float_info.max
        last_valid_accuracy = .0

        # step 0. prepare tensors for collecting hard examples
        top_k_values, top_k_indices = tf.nn.top_k(self.model.each_loss, k=self.batch_size)
        hard_example_images = tf.gather(self.model.input_image_placeholder, top_k_indices)
        hard_example_labels = tf.gather(self.model.label_placeholder, top_k_indices)

        while True:
            start_time = time.time()

            batch_size_forward_only = self.hard_sampling_factor * self.batch_size
            batch_data = self.train_loader.get_batch(batch_size=batch_size_forward_only)
            if batch_data is None:
                print('%d epoch complete' % self.epoch_counter)
                self.epoch_counter += 1
                continue

            # Step 1. collect hard examples by prediction only pass.
            sess_output = self.sess.run(
                fetches=[
                    hard_example_images,
                    hard_example_labels,
                    self.model.accum_cls_loss,
                    self.model.correct_count,
                    self.model.conf_matrix,
                ],
                feed_dict={
                    self.model.keep_prob_placeholder: 1.0,
                    self.model.is_training_placeholder: False,
                    self.model.input_image_placeholder: batch_data.images,
                    self.model.label_placeholder: batch_data.labels,
                    self.model.batch_size_ph: batch_size_forward_only,
                }
            )

            # Step 2. update training status
            accum_loss += sess_output[2]
            accum_correct_count += sess_output[3]
            if accum_conf_matrix is None:
                accum_conf_matrix = sess_output[4]
            else:
                accum_conf_matrix += sess_output[4]

            # Step 3. train by hard examples
            sess_input = [
                self.model.train_op_disc,
                self.model.global_step_disc,
            ]
            sess_output = self.sess.run(
                fetches=sess_input,
                feed_dict={
                    self.model.lr_placeholder: self.lr,
                    self.model.keep_prob_placeholder: 0.6,
                    self.model.is_training_placeholder: True,
                    self.model.input_image_placeholder: sess_output[0],
                    self.model.label_placeholder: sess_output[1],
                    self.model.batch_size_ph: self.batch_size,
                }
            )
            cur_step = sess_output[-1]

            if cur_step > 0 and cur_step % self.train_log_interval == 0:

                duration = time.time() - start_time
                loss = accum_loss / self.train_log_interval
                accuracy = accum_correct_count / (batch_size_forward_only * self.train_log_interval)

                print("[step %d] training loss = %f, accuracy = %.4f (%.4f sec)" % (cur_step, loss, accuracy, duration))

                # log for tensorboard
                custom_summaries = [
                    tf.Summary.Value(tag='loss', simple_value=loss),
                    tf.Summary.Value(tag='accuracy', simple_value=accuracy),
                    tf.Summary.Value(tag='learning rate', simple_value=self.lr),
                ]
                self.train_summary_writer.add_summary(tf.Summary(value=custom_summaries), cur_step)
                self.train_summary_writer.flush()

                # reset local accumulations
                accum_loss = .0
                accum_correct_count = .0
                accum_conf_matrix = None

            if cur_step > 0 and cur_step % self.valid_log_interval == 0:
                cur_valid_loss, cur_valid_accuracy, valid_conf_matrix = self.valid()
                print("... validation loss = %f, accuracy = %.4f" % (cur_valid_loss, cur_valid_accuracy))
                print("==== confusion matrix ====")
                print(valid_conf_matrix)

                if cur_valid_loss < last_valid_loss:
                    # Save confusion matrix to disk
                    report_file_path = os.path.join(self.log_dirpath, "report_%d" % cur_step+".txt")
                    self.make_report(
                        report_file_path,
                        valid_conf_matrix,
                        cur_valid_loss,
                        cur_valid_accuracy
                    )

                    # Save the variables to disk.
                    ckpt_file_path = self.checkpoint_filepath+"_loss_%.6f"%cur_valid_loss+"_accuracy_%.4f"%cur_valid_accuracy
                    save_path = self.saver.save(
                        self.sess,
                        ckpt_file_path,
                        global_step=cur_step,
                    )
                    print("Model saved in file: %s" % save_path)
                    last_valid_loss, last_valid_accuracy = cur_valid_loss, cur_valid_accuracy

            if cur_step > 0 and cur_step % self.lr_decay_interval == 0:
                self.lr *= self.lr_decay_ratio
                print("\tlearning rate decayed to %f" % self.lr)

        return


def main(argv):
    if FLAGS.use_ohem:
        classifier = ClassifierOHEM(FLAGS.hard_sampling_factor)
    else:
        classifier = Classifier()

    if not FLAGS.valid_only:
        classifier.train()
    else:
        loss, accuracy, accum_conf_matrix = classifier.valid()
        print(">> Validation result")
        print("\tloss = %f"%loss)
        print("\taccuracy = %f"%accuracy)
        print("\t==== confusion matrix ====")
        print(accum_conf_matrix)

    return

if __name__ == '__main__':
    tf.app.run()


