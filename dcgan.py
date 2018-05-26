import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def xavier_init(size):
    xavier_stddev = 1.0 / tf.sqrt(size[0] / 2.0)
    return tf.random_normal(shape = size, stddev = xavier_stddev)


X = tf.placeholder(tf.float32, shape = [None, 64, 64, 3])
Z = tf.placeholder(tf.float32, shape = [None, 100])


def sample_Z(m, n):
    return np.random.uniform(-1.0, 1.0, size = [m, n])


def leaky_relu(x, leak = 0.2):
    return tf.maximum(x, x * leak)


def generator(z):
    with tf.variable_scope('g', reuse = tf.AUTO_REUSE):
        g = tcl.fully_connected(z, 4 * 4 * 1024, activation_fn = tf.nn.relu, normalizer_fn = tcl.batch_norm)
        g = tf.reshape(g, (-1, 4, 4, 1024))
        g = tcl.conv2d_transpose(g, 512, kernel_size = (5, 5), stride = 2, padding = 'SAME', activation_fn = tf.nn.relu, normalizer_fn = tcl.batch_norm)
        g = tcl.conv2d_transpose(g, 256, kernel_size = (5, 5), stride = 2, padding = 'SAME', activation_fn = tf.nn.relu, normalizer_fn = tcl.batch_norm)
        g = tcl.conv2d_transpose(g, 128, kernel_size = (5, 5), stride = 2, padding = 'SAME', activation_fn = tf.nn.relu, normalizer_fn = tcl.batch_norm)
        G_prob = tcl.conv2d_transpose(g, 1, kernel_size = (5, 5), stride = 2, padding = 'SAME', activation_fn = tf.nn.tanh)

    return G_prob, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'g')


def discriminator(x):
    with tf.variable_scope('d', reuse =  tf.AUTO_REUSE) :
        d = tcl.conv2d(x, 64, kernel_size = (5, 5), stride = 2, padding = 'SAME', activation_fn = leaky_relu)
        d = tcl.conv2d(d, 128, kernel_size = (5, 5), stride = 2, padding = 'SAME', activation_fn = leaky_relu, normalizer_fn = tcl.batch_norm)
        d = tcl.conv2d(d, 256, kernel_size = (5, 5), stride = 2, padding = 'SAME', activation_fn = leaky_relu, normalizer_fn = tcl.batch_norm)
        d = tcl.conv2d(d, 512, kernel_size = (5, 5), stride = 2, padding = 'SAME', activation_fn = leaky_relu, normalizer_fn = tcl.batch_norm)
        d = tcl.flatten(d)
        D_logit = tcl.fully_connected(d, 1, activation_fn = None)
        D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'd')


def plot(samples):
    fig = plt.figure(figsize = (4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace = 0.05, hspace = 0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis = 'off'
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        #plt.imshow(sample.reshape(28, 28), cmap = 'Greys_r')
        plt.imshow(sample, cmap = 'Greys_r')

    return fig


G_sample, theta_G = generator(Z)

D_real, D_logit_real, theta_D = discriminator(X)
D_fake, D_logit_fake, theta_D = discriminator(G_sample)

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1.0 - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list = theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list = theta_G)

mb_size = 128
Z_dim = 100

mnist = input_data.read_data_sets('../../MNIST_data', one_hot = True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(1000000):
    if it % 1000 == 0:
        samples = sess.run(G_sample, feed_dict = {Z: sample_Z(16, Z_dim)})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches = 'tight')
        i += 1
        plt.close(fig)

    X_mb, _ = mnist.train.next_batch(mb_size)
    X_mb = X_mb.reshape(mb_size, 28, 28, 1)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict = {X: X_mb, Z: sample_Z(mb_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict = {Z: sample_Z(mb_size, Z_dim)})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D_loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()

