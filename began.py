import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from data.load import mnist
from nets.mlp import *
from utils import *


class be_gan:
    def __init__(self, generator, discriminator, data, sess, height = 28, width = 28, iters = 1000000):
        self.x_dim = height * width
        self.y_dim = 10
        self.z_dim = 100
        self.h_dim = 128

        self.batch_size = 32
        self.learning_rate = 1e-4
        self.iters = iters

        self.lam = 1e-3
        self.gamma = 0.5

        self.sess = sess

        self.generator = generator
        self.discriminator = discriminator
        self.data = data

        self.generator.set(self.x_dim, self.h_dim)
        self.discriminator.set(self.x_dim, self.h_dim)
        self.build_model()


    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape = [None, self.x_dim])
        self.z = tf.placeholder(tf.float32, shape = [None, self.z_dim])
        self.k = tf.placeholder(tf.float32)

        self.g_sample = self.generator(self.z)
        self.d_real = self.discriminator(self.x)
        self.d_fake = self.discriminator(self.g_sample, reuse = True)

        self.d_loss = d_real - k * d_fake
        self.g_loss = d_fake


    def train(self):
        d_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.d_loss, var_list = self.discriminator.vars)
        g_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.g_loss, var_list = self.generator.vars)

        self.sess.run(tf.global_variables_initializer())

        if not os.path.exists('out/'):
            os.makedirs('out/')

        k_next = 0
        i = 0
        for it in range(self.iters):
            if it % 1000 == 0:
                samples = sess.run(self.g_sample, feed_dict = {self.z: sample_z(16, self.z_dim)})
                fig = data2img(samples)
                plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches = 'tight')
                i += 1
                plt.close(fig)
    
            x_batch, _ = self.data.next_batch(self.batch_size)
            x_batch = np.reshape(x_batch, (-1, self.x_dim))

            _, d_real, d_fake = sess.run([d_solver, self.d_real, self.d_fake], feed_dict = {self.x: x_batch, self.z: sample_z(self.batch_size, self.z_dim), self.k: k_next})
            sess.run([g_solver], feed_dict = {self.z: sample_z(self.batch_size, self.z_dim)})
            k_next = self.k + self.lam * (self.gamma * d_real - d_fake)

            if it % 1000 == 0:
                convergence = d_real + np.abs(self.gamma * d_real - d_fake)
                print('Iter: {}'.format(it))
                print('convergence: {:.4}'. format(convergence))
                print()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    generator = g_mlp_mnist()
    discriminator = d_mlp_autoencoder_mnist()
    data = mnist()

    sess = tf.Session()

    gan = be_gan(generator, discriminator, data, sess)
    gan.train()

