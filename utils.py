import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def xavier_init(size):
    xavier_stddev = 1. / tf.sqrt(size[0] / 2.)
    return tf.random_normal(shape = size, stddev = xavier_stddev)


def data2img(samples, is_mnist = True):
    fig = plt.figure(figsize = (4, 4)) 
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace = 0.05, hspace = 0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis = 'off'
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if is_mnist:
            plt.imshow(sample.reshape(28, 28), cmap = 'Greys_r')
        else:
            plt.imshow(sample)

    return fig


def sample_z(m, n): 
    return np.random.uniform(-1.0, 1.0, size = [m, n])

