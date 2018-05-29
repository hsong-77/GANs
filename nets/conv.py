import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl


class g_conv():
    def __init__(self):
        self.name = 'g_conv'


    def __call__(self, z):
        with tf.variable_scope(self.name):
            g = tcl.fully_connected(z, self.size * self.size * self.channels[0], activation_fn = tf.nn.relu, normalizer_fn = tcl.batch_norm)
            g = tf.reshape(g, [self.size, self.size, self.channels[0]])
            g = tcl.conv2d_transpose(g, self.channels[1], kernel_size = (5, 5), stride = 2, padding = 'SAME', activation_fn = tf.nn.relu, normalizer_fn = tcl.batch_norm)
            g = tcl.conv2d_transpose(g, self.channels[2], kernel_size = (5, 5), stride = 2, padding = 'SAME', activation_fn = tf.nn.relu, normalizer_fn = tcl.batch_norm)
            g = tcl.conv2d_transpose(g, self.channels[3], kernel_size = (5, 5), stride = 2, padding = 'SAME', activation_fn = tf.nn.relu, normalizer_fn = tcl.batch_norm)
            g = tcl.conv2d_transpose(g, 3, kernel_size = (5, 5), stride = 2, padding = 'SAME', activation_fn = tf.nn.tanh)

        return g


    def set(self, channels = [1024, 512, 256, 128], size = 4):
        self.channels = channels
        self.size = size


    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)


class d_conv():
    def __init__(self):
        self.name = 'd_conv'


    def __call__(self, x, reuse = False):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            d = tcl.conv2d(x, self.channels[0], kernel_size = (5, 5), stride = 2, padding = 'SAME', activation_fn = _leaky_relu)
            d = tcl.conv2d(d, self.channels[1], kernel_size = (5, 5), stride = 2, padding = 'SAME', activation_fn = leaky_relu, normalizer_fn = tcl.batch_norm)
            d = tcl.conv2d(d, self.channels[2], kernel_size = (5, 5), stride = 2, padding = 'SAME', activation_fn = leaky_relu, normalizer_fn = tcl.batch_norm)
            d = tcl.conv2d(d, self.channels[3], kernel_size = (5, 5), stride = 2, padding = 'SAME', activation_fn = leaky_relu, normalizer_fn = tcl.batch_norm)
            d = tcl.flatten(d)
            d_logit = tcl.fully_connected(d, 1, activation_fn = None)
            d_prob = tf.nn.sigmoid(D_logit)

        return d_prob, d_logit


    def set(self, channels = [64, 128, 256, 512]):
        self.channels = channels


    def _leaky_relu(x, leak = 0.2):
        return tf.maximum(x, x * leak)


    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)

