import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from data.load import celeba
from nets.conv import *
from utils import *


class eb_gan:
    def __init__(self, generator, discriminator, data, sess, height = 64, width = 64, iters = 1000000):
        self.height = height
        self.width = width
        self.z_dim = 100
        self.g_channels = [1024, 512, 256, 128]
        self.d_channels = [64, 128, 256, 512, 1024]
        self.margin = 10
        self.pt_loss_weight = 0.1

        self.batch_size = 32
        self.learning_rate = 1e-4
        self.iters = iters

        self.sess = sess

        self.generator = generator
        self.discriminator = discriminator
        self.data = data

        self.generator.set(self.g_channels)
        self.discriminator.set(self.d_channels)
        self.build_model()


    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape = [None, self.height, self.width, 3])
        self.z = tf.placeholder(tf.float32, shape = [None, self.z_dim])

        self.g_sample = self.generator(self.z)
        d_real = self.discriminator(self.x)
        d_fake = self.discriminator(self.g_sample, reuse = True)

        d_fake_loss = self._mse_loss(self.g_sample, d_fake)
        self.d_loss = self._mse_loss(self.x, d_real) + tf.maximum(self.margin - d_fake_loss, 0)
        self.g_loss = d_fake_loss + self.pt_loss_weight * self._pullaway_loss(d_fake)


    def train(self):
        d_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.d_loss, var_list = self.discriminator.vars)
        g_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.g_loss, var_list = self.generator.vars)

        self.sess.run(tf.global_variables_initializer())

        if not os.path.exists('out/'):
            os.makedirs('out/')

        i = 0
        for it in range(self.iters):
            print(it)
            if it % 1000 == 0:
                samples = sess.run(self.g_sample, feed_dict = {self.z: sample_z(16, self.z_dim)})
                fig = data2img(samples, False)
                plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches = 'tight')
                i += 1
                plt.close(fig)
    
            x_batch = self.data.next_batch(self.batch_size)

            _, d_loss_curr = sess.run([d_solver, self.d_loss], feed_dict = {self.x: x_batch, self.z: sample_z(self.batch_size, self.z_dim)})
            _, g_loss_curr = sess.run([g_solver, self.g_loss], feed_dict = {self.z: sample_z(self.batch_size, self.z_dim)})

            if it % 1000 == 0:
                print('Iter: {}'.format(it))
                print('d_loss: {:.4}'. format(d_loss_curr))
                print('g_loss: {:.4}'. format(g_loss_curr))
                print()


    def _mse_loss(self, emb):
        return tf.reduce_mean(tf.reduce_sum((data - emb) ** 2, axis = 1))


    def _pullaway_loss(self, emb):
        norm = tf.sqrt(tf.reduce_sum(emb ** 2, 1, keep_dims = True))
        normalized_emb = emb / norm
        similarity = tf.matmul(normalized_emb, normalized_emb, transpose_b = True)
        batch_size = tf.cast(tf.shape(emb)[0], tf.float32)
        pt_loss = (tf.reduce_sum(similarity) - batch_size) / (batch_size * (batch_size - 1))

        return pt_loss


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    generator = g_conv()
    discriminator = d_conv_autoencoder()
    data = celeba()

    sess = tf.Session()

    gan = dc_gan(generator, discriminator, data, sess)
    gan.train()

