from __future__ import division
import os
import time
from datetime import datetime
from glob import glob
import tensorflow as tf
from six.moves import xrange
from scipy.misc import imresize
from subpixel import PS

from ops import *
from utils import *


def doresize(x, shape):
    x = np.copy((x + 1.) * 127.5).astype("uint8")
    y = imresize(x, shape)
    return y


class DCGAN(object):
    def __init__(self,sess,
                 image_size,
                 batch_size,
                 image_shape=[128, 128, 3],
                 y_dim=None,
                 z_dim=100,
                 gf_dim=64,
                 df_dim=64,
                 gfc_dim=1024,
                 dfc_dim=1024,
                 dataset=None,
                 checkpoint_dir=None,
                 checkpoint_load_dir=None):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
        """
        self.sess = sess
        self.batch_size = batch_size
        self.image_size = image_size
        self.input_size = 32
        self.sample_size = batch_size
        self.image_shape = image_shape
        self.dataset = dataset

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_load_dir = checkpoint_load_dir
        today = datetime.today()
        time_now = "%s%s_%s%s" % (str(today.month).zfill(2),
                                  str(today.day).zfill(2),
                                  str(today.hour).zfill(2),
                                  str(today.minute).zfill(2))
        self.model_dir = "%s_%s_%s" % (self.dataset, self.batch_size, time_now)

        self.build_model()

    def build_model(self):

        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.input_size, self.input_size, 3],
                                    name='real_images')
        try:
            self.up_inputs = tf.image.resize_images(self.inputs,
                                                    self.image_shape[0],
                                                    self.image_shape[1],
                                                    tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        except ValueError:
            # newer versions of tensorflow
            self.up_inputs = tf.image.resize_images(self.inputs,
                                                    [self.image_shape[0], self.image_shape[1]],
                                                    tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        self.images = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape,
                                    name='real_images')
        self.sample_images= tf.placeholder(tf.float32, [self.sample_size] + self.image_shape,
                                        name='sample_images')

        self.G = self.generator(self.inputs)
        self.d_real, self.d_real_logits = self.discriminator(self.images)
        self.d_fake, self.d_fake_logits = self.discriminator(self.G, reuse=True)

        self.G_sum = tf.summary.image("G", self.G)
        self.d_real_sum = tf.summary.histogram("d_real", self.d_real)
        self.d_fake_sum = tf.summary.histogram("d_fake", self.d_fake)

        mse = tf.losses.mean_squared_error(self.images, self.G, weights=1.0)
        #psnr = tf.reduce_mean(tf.image.psnr(self.images, self.G, max_val=255))
        ssim = tf.reduce_mean(tf.image.ssim(self.images, self.G, max_val=1.0))

        self.d_loss_real = tf.reduce_mean(
            self.sigmoid_cross_entropy_with_logits(self.d_real_logits, tf.ones_like(self.d_real)))
        self.d_loss_fake = tf.reduce_mean(
            self.sigmoid_cross_entropy_with_logits(self.d_fake_logits, tf.zeros_like(self.d_fake)))

        self.g_loss = 100 * mse - ssim

           # self.g_loss_info = "sigmoid"
           # self.g_loss = tf.reduce_mean(
           #     sigmoid_cross_entropy_with_logits(self.d_fake_logits, tf.ones_like(self.d_fake)))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.mse = tf.summary.scalar("mse", mse)
        #self.psnr = tf.summary.scalar("psnr", psnr)
        self.ssim = tf.summary.scalar("ssim", ssim)

        t_vars = tf.trainable_variables()
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.saver = tf.train.Saver()

    def sigmoid_cross_entropy_with_logits(self, x, y):
        try:
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
        except:
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    def train(self, config):
        """Train DCGAN"""
        # first setup validation data
        data = sorted(glob(os.path.join("./dataset", config.dataset, "valid", "*.jpg")))

        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)

        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        self.g_sum = tf.summary.merge(
            [self.G_sum, self.g_loss_sum, self.mse, self.ssim])
        self.d_sum = tf.summary.merge(
            [self.d_real_sum, self.d_fake_sum, self.d_loss_real_sum, self.d_loss_fake_sum])
        self.writer = tf.summary.FileWriter(os.path.join("./logs",  self.model_dir), self.sess.graph)

        sample_files = data[0:self.sample_size]
        sample = [get_image(sample_file) for sample_file in sample_files]
        sample_inputs = [doresize(xx, [self.input_size,]*2) for xx in sample]
        sample_images = np.array(sample).astype(np.float32)
        sample_input_images = np.array(sample_inputs).astype(np.float32)


        if not os.path.exists("./samples/" + self.model_dir):
            os.mkdir(os.path.join("./samples", self.model_dir))

        # loss_log_file = open(os.path.join("./samples", self.model_dir, "g_loss.log"), "w+")

        save_images(sample_input_images, [8, 8], os.path.join("./samples", self.model_dir, 'inputs_small.png'))
        save_images(sample_images, [8, 8], os.path.join("./samples", self.model_dir, 'reference.png'))

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_load_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # we only save the validation inputs once
        have_saved_inputs = False

        for epoch in xrange(config.epoch):
            data = sorted(glob(os.path.join("./dataset", config.dataset, "train", "*.jpg")))
            batch_idxs = min(len(data), config.train_size) // config.batch_size
            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = [get_image(batch_file) for batch_file in batch_files]
                input_batch = [doresize(xx, [self.input_size,]*2) for xx in batch]
                batch_images = np.array(batch).astype(np.float32)
                batch_inputs = np.array(input_batch).astype(np.float32)

                errD = 0.0

                if epoch <= 70:
                    # Update G network
                    _, summary_str, errG = self.sess.run([g_optim, self.g_sum, self.g_loss],
                                                         feed_dict={self.inputs: batch_inputs,
                                                                    self.images: batch_images})

                    self.writer.add_summary(summary_str, counter)

                elif 70 < epoch <= 81:
                    # Update D network
                    _, summary_str, errD = self.sess.run([d_optim, self.d_sum, self.d_loss],
                        feed_dict={self.inputs: batch_inputs, self.images: batch_images})

                    self.writer.add_summary(summary_str, counter)

                else:
                    # Loss Change
                    self.g_loss = tf.reduce_mean(
                        self.sigmoid_cross_entropy_with_logits(self.d_fake_logits, tf.ones_like(self.d_fake)))
                    # Update G network
                    _, summary_str, errG = self.sess.run([g_optim, self.g_sum, self.g_loss],
                                                         feed_dict={self.inputs: batch_inputs,
                                                                    self.images: batch_images})

                    self.writer.add_summary(summary_str, counter)

                    # Update D network
                    _, summary_str, errD = self.sess.run([d_optim, self.d_sum, self.d_loss],
                                                         feed_dict={self.inputs: batch_inputs,
                                                                    self.images: batch_images})

                    self.writer.add_summary(summary_str, counter)

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, g_loss: %.8f, d_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errG, errD))

                if np.mod(counter, 170) == 1:
                    samples, g_loss, up_inputs = self.sess.run(
                        [self.G, self.g_loss, self.up_inputs],
                        feed_dict={self.inputs: sample_input_images, self.images: sample_images}
                    )
                    if not have_saved_inputs:
                        save_images(up_inputs, [8, 8], os.path.join("./samples", self.model_dir, './inputs.png'))
                        have_saved_inputs = True
                    save_images(samples, [8, 8], os.path.join("./samples",
                                                              self.model_dir, "valid_%s_%s.png" % (epoch, idx)))
                    #loss_log_file.write("[Sample] g_loss: %.8f" % g_loss)


                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)
        #loss_log_file.close()

    def generator(self, inp):
        with tf.variable_scope("generator") as scope:
            self.h0, self.h0_w, self.h0_b = deconv2d(inp, [self.batch_size, 32, 32, self.gf_dim], k_h=1, k_w=1, d_h=1, d_w=1,
                                                     name='g_h0', with_w=True)
            h0 = lrelu(self.h0)

            self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, 32, 32, self.gf_dim], name='g_h1', k_h=1, k_w=1, d_h=1, d_w=1,
                                                     with_w=True)
            h1 = lrelu(self.h1)

            self.h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, 32, 32, self.gf_dim], name='g_h2', d_h=1, d_w=1,
                                                     with_w=True)
            h2 = lrelu(self.h2)

            self.h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, 32, 32, 3*16], d_h=1, d_w=1, name='g_h3', with_w=True)
            h3 = lrelu(self.h3)

            h4 = PS(h3, 4, color=True)

            return tf.nn.tanh(h4)

    def discriminator(self, inp, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(inp, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(conv2d(h0, self.df_dim*2, name='d_h1_conv'))
            h2 = lrelu(conv2d(h1, self.df_dim * 4, name='d_h2_conv'))
            h3 = lrelu(conv2d(h2, self.df_dim * 8, name='d_h3_conv'))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

            return tf.nn.sigmoid(h4), h4


    def save(self, checkpoint_dir, step):
        model_name = "subpixel.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_load_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(self.checkpoint_dir, checkpoint_load_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
