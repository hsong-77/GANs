import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from data.load import mnist
from nets.mlp_q import *
from utils import *


class info_gan:
    def __init__(self, generator, discriminator, data, sess, height = 28, width = 28, iters = 1000000):
        self.x_dim = height * width
        self.y_dim = 10
        self.z_dim = 100    #16
        self.h_dim = 128
        self.c_idm = 10

        self.batch_size = 32
        self.learning_rate = 1e-4
        self.iters = iters

        self.sess = sess

        self.generator = generator
        self.discriminator = discriminator
        self.data = data

        self.generator.set(self.x_dim, self.h_dim)
        self.discriminator.set(self.h_dim, self.c_dim)
        self.build_model()


    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape = [None, self.x_dim])
        self.z = tf.placeholder(tf.float32, shape = [None, self.z_dim])
        self.c = tf.placeholder(tf.float32, shape = [None, self.c_dim])

        self.g_sample = self.generator(self.z, self.c)
        d_real, _ = self.discriminator(self.x)
        d_fake, q_fake = self.discriminator(self.g_sample, reuse = True)

        d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_real, labels = tf.ones_like(d_real)))
        d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_fake, labels = tf.zeros_like(d_fake)))
        self.d_loss = d_real_loss + d_fake_loss
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_fake, labels = tf.ones_like(d_fake)))
        self.q_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = q_fake, labels = self.c))


    def train(self):
        d_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.d_loss, var_list = self.discriminator.vars)
        g_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.g_loss, var_list = self.generator.vars)
        q_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.q_loss, var_list = self.generator.vars + self.discriminator.vars)

        self.sess.run(tf.global_variables_initializer())

        if not os.path.exists('out/'):
            os.makedirs('out/')

        i = 0
        for it in range(self.iters):
            if it % 1000 == 0:
                z_noise = sample_z(16, self.z_dim)
                idx = np.random.randint(0, self.c_dim)
                c_noise = np.zeros(16, self.c_dim)
                c_noise[range(16), idx] = 1
                samples = sess.run(self.g_sample, feed_dict = {self.z: z_noise, self.c = c_noise})

                fig = data2img(samples)
                plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches = 'tight')
                i += 1
                plt.close(fig)
    
            x_batch, _ = self.data.next_batch(self.batch_size)
            x_batch = np.reshape(x_batch, (-1, self.x_dim))
            
            _, d_loss_curr = sess.run([d_solver, self.d_loss], feed_dict = {self.x: x_batch, self.z: sample_z(self.batch_size, self.z_dim), self.c: self._sample_c(self.batch_size)})
            _, g_loss_curr = sess.run([g_solver, self.g_loss], feed_dict = {self.z: sample_z(self.batch_size, self.z_dim), self.c: self._sample_c(self.batch_size)})
            sess.run([q_solver], feed_dict = {self.z: sample_z(self.batch_size, self.z_dim), self.c: self._sample_c(self.batch_size)})

            if it % 1000 == 0:
                print('Iter: {}'.format(it))
                print('d_loss: {:.4}'. format(d_loss_curr))
                print('g_loss: {:.4}'. format(g_loss_curr))
                print()


    def _sample_c(m, ind = -1):
        c = np.zeros([m, self.c_dim])
        for i in range(m):
            if ind < 0:
                ind = np.random.randint(self.c_dim)
            c[i, ind] = 1

        return c


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    generator = g_mlp_q_mnist()
    discriminator = d_mlp_q_mnist()
    data = mnist()

    sess = tf.Session()

    gan = vanilla_gan(generator, discriminator, data, sess)
    gan.train()

