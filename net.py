import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl


class g_mlp_mnist():
    def __init__(self):
        self.name = 'g_mlp_mnist'


    def __call__(self, z):
        with tf.variable_scope(self.name):
            g = tcl.fully_connected(z, self.h_dim, activation_fn = tf.nn.relu, weights_initializer = tf.random_normal_initializer(0, 0.02))
            g = tcl.fully_connected(g, self.x_dim, activation_fn = tf.nn.sigmoid, weights_initializer = tf.random_normal_initializer(0, 0.02))

        return g


    def set(self, x_dim, h_dim):
        self.x_dim = x_dim
        self.h_dim = h_dim


    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)


class d_mlp_mnist():
    def __init__(self):
        self.name = 'd_mlp_mnist'


    def __call__(self, x, reuse = False):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            d = tcl.fully_connected(x, self.h_dim, activation_fn = tf.nn.relu, weights_initializer = tf.random_normal_initializer(0, 0.02))
            d = tcl.fully_connected(d, 1, activation_fn = tf.nn.sigmoid, weights_initializer = tf.random_normal_initializer(0, 0.02))

        return d

    def set(self, h_dim):
        self.h_dim = h_dim


    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)

