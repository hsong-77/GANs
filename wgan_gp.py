import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from data.load import celeba
from nets.conv import *
from utils import *


class w_gan_gp:
    def __init__(self, generator, discriminator, data, sess, height = 64, width = 64, iters = 1000000):
        self.height = height
        self.width = width
        self.z_dim = 100
        self.g_channels = [1024, 512, 256, 128]
        self.d_channels = [64, 128, 256, 512]

        self.batch_size = 32
        self.learning_rate = 1e-4
        self.iters = iters
        self.d_iters = 5
        self.lam = 10

        self.sess = sess

        self.generator = generator
        self.discriminator = discriminator
        self.data = data

        self.generator.set(g_channels)
        self.discriminator.set(True, d_channels)
        self.build_model()


    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape = [None, self.height, self.width, 3])
        self.z = tf.placeholder(tf.float32, shape = [None, self.z_dim])

        self.g_sample = self.generator(self.z)
        _, d_real_logit = self.discriminator(self.x)
        _, d_fake_logit = self.discriminator(self.g_sample, reuse = True)

        self.d_loss = -tf.reduce_mean(d_real_logit) + tf.reduce_mean(d_fake_logit)
        self.g_loss = -tf.reduce_mean(d_fake_logit)

        alpha = tf.random_uniform([self.batch_size, 1], minval = 0., maxval = 1.)
        interpolates = alpha * self.x + (1. - alpha) * self.g_sample
        grad = tf.gradients(self.discriminator(interpolates), [interpolates])[0]
        grad_norm = tf.sqrt(tf.reduce_sum(grad ** 2, axis = 1))
        grad_pen = self.lam * tf.reduce_mean((grad_norm - 1.) ** 2)

        self.d_loss += grad_pen


    def train(self):
        d_solver = tf.train.AdamOptimizer(self.learning_rate, beta1 = 0.5, beta2 = 0.9).minimize(self.d_loss, var_list = self.discriminator.vars)
        g_solver = tf.train.AdamOptimizer(self.learning_rate, bata1 = 0.5, beta2 = 0.9).minimize(self.g_loss, var_list = self.generator.vars)

        self.sess.run(tf.global_variables_initializer())

        if not os.path.exists('out/'):
            os.makedirs('out/')

        i = 0
        for it in range(self.iters):
            if it % 1000 == 0:
                samples = sess.run(self.g_sample, feed_dict = {self.z: sample_z(16, self.z_dim)})
                fig = data2img(samples)
                plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches = 'tight')
                i += 1
                plt.close(fig)
    
            for _ in range(self.d_iters):
                x_batch = self.data.next_batch(self.batch_size)
                _, d_loss_curr = sess.run([d_solver, self.d_loss], feed_dict = {self.x: x_batch, self.z: sample_z(self.batch_size, self.z_dim)})
            _, g_loss_curr = sess.run([g_solver, self.g_loss], feed_dict = {self.z: sample_z(self.batch_size, self.z_dim)})

            if it % 1000 == 0:
                print('Iter: {}'.format(it))
                print('d_loss: {:.4}'. format(d_loss_curr))
                print('g_loss: {:.4}'. format(g_loss_curr))
                print()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    generator = g_conv()
    discriminator = d_conv()
    data = celeba()

    sess = tf.Session()

    gan = w_gan(generator, discriminator, data, sess)
    gan.train()

