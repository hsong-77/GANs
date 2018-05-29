import os
import time
import math
import tensorflow as tf
import numpy as np
import scipy.misc
from six.moves import xrange
from glob import glob

from data.config import *


class mnist():
    def __init__(self):
        self.y_dim = 10
        self.__x, self.__y = self.__load_mnist()
        self.__batch_start = 0


    def next_batch(self, batch_size):
        batch_x = []
        batch_y = []

        batch_end = self.__batch_start + batch_size
        data_size = len(self.__x)
        if batch_end < data_size:
            batch_x = self.__x[self.__batch_start : batch_end]
            batch_y = self.__y[self.__batch_start : batch_end]
            self.__batch_start = batch_end
        else:
            batch_x = np.concatenate((self.__x[self.__batch_start : data_size], self.__x[0 : batch_end - data_size]), axis = 0)
            batch_y = np.concatenate((self.__y[self.__batch_start : data_size], self.__y[0 : batch_end - data_size]), axis = 0)
            self.__batch_start = batch_end - data_size

        return batch_x, batch_y


    def __load_mnist(self):
        data_dir = os.path.join(data_dirpath, 'mnist')

        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file = fd, dtype = np.uint8)
        trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file = fd, dtype = np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file = fd, dtype = np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file = fd, dtype = np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        X = np.concatenate((trX, teX), axis = 0)
        y = np.concatenate((trY, teY), axis=0).astype(np.int)

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        y_vec = np.zeros((len(y), self.y_dim), dtype = np.float)
        for i, label in enumerate(y):
            y_vec[i, y[i]] = 1.0

        return X / 255., y_vec


class celeba():
    def __init__(self):
        self.channel = 3
        self.size = 64
        self.__data = glob(os.path.join(data_path, '*.jpg'))
        self.__batch_start = 0


    def next_batch(self, batch_size):
        batch_path = []

        batch_end = self.__batch_start + batch_size
        data_size = len(self.__data)
        if batch_end < data_size:
            batch_path = self.__datra[self.__batch_start : batch_end]
            self.__batch_start = batch_end
        else:
            batch_path = np.concatenate((self.__data[self.__batch_start : data_size], self.__data[0 : batch_end - data_size]), axis = 0)
            self.__batch_start = batch_end - data_size

        batch_image = [get_image(img_path, True, 128, 128, self.size, self.size) for img_path in batch_path]
        batch_image = np.array(batch_image).astype(np.float32)

        return batch_image


def get_image(img_path, is_crop = True, crop_h, crop_w, resize_h, resize_w):
    image = scipy.misc.imread(img_path).astype(np.float)

	if is_crop:
		h, w = image.shape[: 2]
		j = int(round((h - crop_h) / 2.))
		i = int(round((w - crop_w) / 2.))
		cropped_image = scipy.misc.imresize(image[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])
	else:
		cropped_image = scipy.misc.imresize(image, [resize_h, resize_w])

	return np.array(cropped_image) / 127.5 - 1


if __name__ == '__main__':
    print('load...')
