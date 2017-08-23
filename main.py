import os
import shutil
import sys
import time
import tensorflow as tf

from loader import *
from loader_oversampling import *
from model import *
import cv2

tf.app.flags.DEFINE_string('data_path', '../data', 'Directory path to read the data files')
tf.app.flags.DEFINE_string('checkpoint_path', 'model', 'Directory path to save checkpoint files')
tf.app.flags.DEFINE_string('model', 'gan', 'gan / acgan / info-acgan')

tf.app.flags.DEFINE_boolean('train_continue', False, 'flag for continue training from previous checkpoint')
tf.app.flags.DEFINE_boolean('valid_only', False, 'flag for validation only. this will make train_continue flag ignored')

tf.app.flags.DEFINE_integer('batch_size', 128, 'mini-batch size for training')
tf.app.flags.DEFINE_float('lr_disc', 2e-4, 'initial learning rate for discriminator')
tf.app.flags.DEFINE_float('lr_gen', 4e-4, 'initial learning rate for generator')
tf.app.flags.DEFINE_float('lr_decay_ratio', 1.0, 'ratio for decaying learning rate')
tf.app.flags.DEFINE_integer('lr_decay_interval', 10000, 'step interval for decaying learning rate')
tf.app.flags.DEFINE_integer('train_log_interval', 200, 'step interval for triggering print logs of train')
tf.app.flags.DEFINE_integer('valid_log_interval', 1000, 'step interval for triggering validation')

tf.app.flags.DEFINE_boolean('use_ohem', False, 'use OHEM or not')
tf.app.flags.DEFINE_integer('hard_sampling_factor', 2, 'searching window size for OHEM')

tf.app.flags.DEFINE_boolean('use_oversampling', False, 'use over-sampling for loader or not')

FLAGS = tf.app.flags.FLAGS

print("Learning rate for discriminator = %e" % FLAGS.lr_disc)
print("Learning rate for generator = %e" % FLAGS.lr_gen)


class GanLearner:
    def __init__(self):
        self.sess = tf.Session()
        self.batch_size = FLAGS.batch_size
        self.model = ACGANModel(batch_size=self.batch_size)

        if FLAGS.use_oversampling:
            self.train_loader = Cifar10LoaderOversampling(data_path=os.path.join(FLAGS.data_path, "train"), batch_size=self.batch_size)
        else:
            self.train_loader = OcrLoader(data_path=os.path.join(FLAGS.data_path, "train"), batch_size=self.batch_size)
        self.valid_loader = OcrLoader(data_path=os.path.join(FLAGS.data_path, "val"), batch_size=self.batch_size)

        self.epoch_counter = 1
        self.lr_gen = FLAGS.lr_gen
        self.lr_disc = FLAGS.lr_disc
        self.lr_decay_interval = FLAGS.lr_decay_interval
        self.lr_decay_ratio = FLAGS.lr_decay_ratio
        self.train_log_interval = FLAGS.train_log_interval
        self.valid_log_interval = FLAGS.valid_log_interval

        self.keep_prob = 0.7

        self.train_continue = FLAGS.train_continue or FLAGS.valid_only
        self.checkpoint_dirpath = FLAGS.checkpoint_path
        self.checkpoint_filepath = os.path.join(self.checkpoint_dirpath, 'model.ckpt')
        self.log_dirpath = "log"

        if not self.train_continue and os.path.exists(self.checkpoint_dirpath):
            shutil.rmtree(self.log_dirpath, ignore_errors=True)
            shutil.rmtree(self.checkpoint_dirpath, ignore_errors=True)
            shutil.rmtree("fake_images", ignore_errors=True)
        if not os.path.exists(self.checkpoint_dirpath):
            os.makedirs(self.checkpoint_dirpath)
            os.makedirs("fake_images")

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

        last_disc_loss_real = sys.float_info.max
        last_real_cls_loss = sys.float_info.max

        last_disc_loss_fake = sys.float_info.max
        last_fake_cls_loss = sys.float_info.max

        last_disc_loss = sys.float_info.max
        last_gen_loss = sys.float_info.max

        accum_correct_count = .0
        accum_conf_matrix = None

        last_valid_disc_loss = sys.float_info.max
        last_valid_gen_loss = sys.float_info.max
        last_valid_accuracy = .0

        cur_step = 0

        summaries = tf.summary.merge_all()

        while True:
            start_time = time.time()
            batch_data = self.train_loader.get_batch()
            if batch_data is None:
                print('%d epoch complete' % self.epoch_counter)
                self.epoch_counter += 1
                continue

            # Step 1. train for discriminator
            sess_output = self.sess.run(
                fetches=[
                    self.model.train_op_disc,
                    self.model.disc_loss,
                    self.model.gen_loss,
                    summaries,
                ],
                feed_dict={
                    self.model.lr_disc_ph: self.lr_disc,
                    self.model.keep_prob_ph: self.keep_prob,
                    self.model.is_training_gen_ph: False,
                    self.model.is_training_disc_ph: True,
                    self.model.input_image_ph: batch_data.images,
                    self.model.label_cls_ph: batch_data.labels,
                    self.model.batch_size_ph: self.batch_size,
                }
            )
            last_disc_loss = sess_output[1]
            last_gen_loss = sess_output[2]
            last_summaries = sess_output[-1]
            print("\tTrain disc. >> disc. = %f, gen. = %f" % (last_disc_loss, last_gen_loss))

            # Step 2. train for generator
            sess_output = self.sess.run(
                fetches=[
                    self.model.train_op_gen,
                    self.model.disc_loss,
                    self.model.gen_loss,
                    self.model.most_realistic_fake_image,
                    self.model.most_realistic_fake_class,
                    summaries,
                    ],
                feed_dict={
                    self.model.lr_gen_ph: self.lr_gen,
                    self.model.keep_prob_ph: self.keep_prob,
                    self.model.is_training_gen_ph: True,
                    self.model.is_training_disc_ph: False,
                    self.model.input_image_ph: batch_data.images,
                    self.model.label_cls_ph: batch_data.labels,
                    self.model.batch_size_ph: self.batch_size,
                }
            )
            last_disc_loss = sess_output[1]
            last_gen_loss = sess_output[2]
            fake_label = sess_output[4][0]
            print("\tTrain gen.  >> disc. = %f, gen. =%f" % (last_disc_loss, last_gen_loss))

            if cur_step > 0 and cur_step % self.train_log_interval == 0:
                cv2.imwrite(
                    "fake_images/step_%d_%s_train.jpg" % (cur_step, self.valid_loader.label_name[fake_label]),
                    (sess_output[3][0] + 1.0) * (255.0/2.0)
                )
            last_summaries = sess_output[-1]

            # NOTE evaluation losses
            sess_input = [
                self.model.gen_loss,
                self.model.disc_loss_real,
                self.model.disc_loss_fake,
                self.model.disc_loss,
                # self.model.cls_loss_real,
                # self.model.cls_loss_fake,
                self.model.correct_count,
                self.model.conf_matrix,
            ]
            sess_output = self.sess.run(
                fetches=sess_input,
                feed_dict={
                    self.model.keep_prob_ph: self.keep_prob,
                    self.model.is_training_gen_ph: False,
                    self.model.is_training_disc_ph: False,
                    self.model.input_image_ph: batch_data.images,
                    self.model.label_cls_ph: batch_data.labels,
                    self.model.batch_size_ph: self.batch_size,
                }
            )
            last_gen_loss = sess_output[0]
            last_disc_loss_real = sess_output[1]
            last_disc_loss_fake = sess_output[2]
            last_disc_loss = sess_output[3]
            # last_real_cls_loss = sess_output[3]
            # last_fake_cls_loss = sess_output[4]

            accum_correct_count += sess_output[-2]
            if accum_conf_matrix is None:
                accum_conf_matrix = sess_output[-1]
            else:
                accum_conf_matrix += sess_output[-1]

            cur_step += 1

            if cur_step > 0 and cur_step % self.train_log_interval == 0:
                duration = time.time() - start_time
                # interval = self.train_log_interval / 2
                interval = 1
                disc_loss_real = last_disc_loss_real / interval
                disc_loss_fake = last_disc_loss_fake / interval
                # cls_loss_real = last_real_cls_loss / interval
                # cls_loss_fake = last_fake_cls_loss / interval
                gen_loss = last_gen_loss / interval
                disc_loss = last_disc_loss / interval
                accuracy = accum_correct_count / (self.batch_size * self.train_log_interval)

                print("[step %d] training loss : disc = %.4f, disc_real = %.4f, disc_fake = %.4f, gen = %f"
                      % (cur_step, disc_loss, disc_loss_real, disc_loss_fake, gen_loss))

                # NOTE summaries for tensorboard
                custom_summaries = [
                    tf.Summary.Value(tag='disc_loss_real', simple_value=disc_loss_real),
                    tf.Summary.Value(tag='disc_loss_fake', simple_value=disc_loss_fake),
                    # tf.Summary.Value(tag='cls_loss_real', simple_value=cls_loss_real),
                    # tf.Summary.Value(tag='cls_loss_fake', simple_value=cls_loss_fake),
                    tf.Summary.Value(tag='disc_loss', simple_value=disc_loss),
                    tf.Summary.Value(tag='gen_loss', simple_value=gen_loss),
                    tf.Summary.Value(tag='accuracy', simple_value=accuracy),
                    tf.Summary.Value(tag='learning rate for discriminator', simple_value=self.lr_disc),
                    tf.Summary.Value(tag='learning rate for generator', simple_value=self.lr_gen),
                ]
                self.train_summary_writer.add_summary(tf.Summary(value=custom_summaries), cur_step)
                self.train_summary_writer.add_summary(last_summaries)
                self.train_summary_writer.flush()

                # reset local accumulations
                accum_correct_count = .0
                accum_conf_matrix = None

            if cur_step > 0 and cur_step % self.valid_log_interval == 0:
                disc_loss_real, disc_loss_fake, gen_loss = self.valid(step=cur_step)

                # print("\t... validation loss : disc_real = %.4f, cls_real = %.4f, disc_fake = %.4f, cls_fake = %.4f, gen = %f, accuracy = %.4f"
                #       % (disc_loss_real, cls_loss_real, disc_loss_fake, cls_loss_fake, gen_loss, accuracy))

                print("\t... validation loss : disc_real = %.4f, disc_fake = %.4f, gen = %f"
                      % (disc_loss_real, disc_loss_fake, gen_loss))

                disc_loss = disc_loss_real + disc_loss_fake

                # print("==== confusion matrix ====")
                # print(valid_conf_matrix)

                if disc_loss < last_valid_disc_loss and gen_loss < last_valid_gen_loss:
                    # Save confusion matrix to disk
                    report_file_path = os.path.join(self.log_dirpath, "report_%d" % cur_step+".txt")
                    # self.make_report(
                    #     report_file_path,
                    #     valid_conf_matrix,
                    #     disc_loss,
                    #     gen_loss,
                    #     accuracy
                    # )

                    # Save the variables to disk.
                    save_path = self.saver.save(
                        self.sess,
                        self.checkpoint_filepath+"_dl_%.4f"%disc_loss+"_gl_%.4f"%gen_loss,
                        global_step=cur_step
                    )
                    print("Model saved in file: %s" % save_path)
                    last_valid_disc_loss = disc_loss
                    last_valid_gen_loss = gen_loss
                    # last_valid_accuracy = accuracy

            if cur_step > 0 and cur_step % self.lr_decay_interval == 0:
                self.lr_disc *= self.lr_decay_ratio
                self.lr_gen *= self.lr_decay_ratio
            if cur_step % self.lr_decay_interval == 0:
                print("\t===> learning rate decayed to %f, %f" % (self.lr_disc, self.lr_gen))

        return

    def valid(self, step=None):
        self.valid_loader.reset()

        step_counter = 0
        accum_disc_loss_real = .0
        accum_cls_loss_real = .0

        accum_disc_loss_fake = .0
        accum_fake_cls_loss = .0
        accum_gen_loss = .0
        accum_disc_loss = .0

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

            # Validation for real & fake samples concurrently.
            sess_input = [
                self.model.gen_loss,
                self.model.disc_loss_fake,
                self.model.disc_loss_real,
                self.model.disc_loss,
                self.model.correct_count,
                self.model.most_realistic_fake_image,
                self.model.most_realistic_fake_class,
            ]
            sess_output = self.sess.run(
                fetches=sess_input,
                feed_dict={
                    self.model.keep_prob_ph: self.keep_prob,
                    self.model.is_training_gen_ph: False,
                    self.model.is_training_disc_ph: False,
                    self.model.input_image_ph: batch_data.images,
                    self.model.label_cls_ph: batch_data.labels,
                    self.model.batch_size_ph: valid_batch_size,
                }
            )
            accum_gen_loss += sess_output[0]
            accum_disc_loss_fake += sess_output[1]
            accum_disc_loss_real += sess_output[2]
            accum_disc_loss += sess_output[3]
            accum_correct_count += sess_output[4]

            fake_label = sess_output[-1][0]
            fake_image = (sess_output[-2][0] + 1.0) * (255.0/2.0)

            step_counter += 1

        # global_step = self.sess.run(self.model.global_step_disc)
        cv2.imwrite("fake_images/step_%d_%s_valid.jpg" % (step, self.valid_loader.label_name[fake_label]), fake_image)

        disc_loss_real = accum_disc_loss_real / step_counter
        disc_loss_fake = accum_disc_loss_fake / step_counter
        # cls_loss_real = accum_cls_loss_real / step_counter
        # cls_loss_fake = accum_fake_cls_loss / step_counter
        gen_loss = accum_gen_loss / step_counter
        disc_loss = accum_disc_loss /step_counter
        accuracy = accum_correct_count / (valid_batch_size * step_counter)

        # log for tensorboard
        cur_step = self.sess.run(self.model.global_step_disc)
        custom_summaries = [
            tf.Summary.Value(tag='disc_loss_real', simple_value=disc_loss_real),
            tf.Summary.Value(tag='disc_loss_fake', simple_value=disc_loss_fake),
            # tf.Summary.Value(tag='cls_loss_real', simple_value=cls_loss_real),
            # tf.Summary.Value(tag='cls_loss_fake', simple_value=cls_loss_fake),
            tf.Summary.Value(tag='gen_loss', simple_value=gen_loss),
            tf.Summary.Value(tag='disc_loss', simple_value=disc_loss),
            tf.Summary.Value(tag='accuracy', simple_value=accuracy),
        ]
        self.valid_summary_writer.add_summary(tf.Summary(value=custom_summaries), cur_step)
        self.valid_summary_writer.flush()

        # return disc_loss_real, cls_loss_real, disc_loss_fake, cls_loss_fake, gen_loss, accuracy, accum_conf_matrix
        return disc_loss_real, disc_loss_fake, gen_loss

def main(argv):

    learner = GanLearner()

    if not FLAGS.valid_only:
        learner.train()
    else:
        loss, accuracy, accum_conf_matrix = learner.valid()
        print(">> Validation result")
        print("\tloss = %f" % loss)
        print("\taccuracy = %f" % accuracy)
        print("\t==== confusion matrix ====")
        print(accum_conf_matrix)

    return

if __name__ == '__main__':
    tf.app.run()


