import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from data.load import mnist
from nets.mlp import *
from utils import *


class vanilla_gan:
    def __init__(self, generator, discriminator, data, sess, height = 28, width = 28, iters = 1000000):
        self.x_dim = height * width
        self.y_dim = 10
        self.z_dim = 100
        self.h_dim = 128

        self.batch_size = 128
        self.learning_rate = 1e-4
        self.iters = iters

        self.sess = sess

        self.generator = generator
        self.discriminator = discriminator
        self.data = data

        self.generator.set(self.x_dim, self.h_dim)
        self.discriminator.set(self.h_dim)
        self.build_model()


    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape = [None, self.x_dim])
        self.z = tf.placeholder(tf.float32, shape = [None, self.z_dim])

        self.g_sample = self.generator(self.z)
        _, d_real_logit = self.discriminator(self.x)
        _, d_fake_logit = self.discriminator(self.g_sample, reuse = True)

        self.d_loss = 0.5 * tf.reduce_mean((d_real_logit - 1) ** 2) + 0.5 * tf.reduce_mean(d_fake_logit ** 2)
        self.g_loss = 0.5 * tf.reduce_mean((d_fake_logit - 1) ** 2)


    def train(self):
        d_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.d_loss, var_list = self.discriminator.vars)
        g_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.g_loss, var_list = self.generator.vars)

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
    
            x_batch, _ = self.data.next_batch(self.batch_size)
            x_batch = np.reshape(x_batch, (-1, self.x_dim))

            _, d_loss_curr = sess.run([d_solver, self.d_loss], feed_dict = {self.x: x_batch, self.z: sample_z(self.batch_size, self.z_dim)})
            _, g_loss_curr = sess.run([g_solver, self.g_loss], feed_dict = {self.z: sample_z(self.batch_size, self.z_dim)})

            if it % 1000 == 0:
                print('Iter: {}'.format(it))
                print('d_loss: {:.4}'. format(d_loss_curr))
                print('g_loss: {:.4}'. format(g_loss_curr))
                print()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    generator = g_mlp_mnist()
    discriminator = d_mlp_mnist()
    data = mnist()

    sess = tf.Session()

    gan = vanilla_gan(generator, discriminator, data, sess)
    gan.train()

