#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 21:06:14 2017

@author: yyang
"""
from __future__ import division

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf
#%% functions to read data and pre-processing
def np_load_csv(fname, add_dummy=False, dummy_col=[0]):
    data = pd.read_csv(fname).values.astype(np.float)
    if add_dummy:
        data = np.insert(data, dummy_col, values=0, axis=1)
    print fname, 'loaded!'
    return data

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data, dtype=int).reshape(-1)
    return np.eye(nb_classes)[targets]

def display_flatten_image(data, img_shape=[28,28]):
    #s = [x for k in [[-1], img_shape] for x in k]
    plt.imshow(data.reshape(img_shape), cmap=cm.binary)


class Dataset(object):
    def __init__(self, data, label=0, one_hot_num=10, scale=1.0):
        idx = [k for k in range(0, data.shape[1]) if k != label]
        x = data[:, idx] / scale
        y = data[:, label]
        if one_hot_num:
            y = indices_to_one_hot(y, one_hot_num)
        self.create(x, y)

    def create(self, x, y):
        self.images = x
        self.labels = y
        self.n = x.shape[0]
        self.shuffle()

    def shuffle(self):
        perm = np.arange(self.n)
        np.random.shuffle(perm)
        self.images = self.images[perm]
        self.labels = self.labels[perm]
        self._next_id = 0

    def next_batch(self, batch_size):
        if self._next_id + batch_size >= self.n:
            self.shuffle()
        cur_id = self._next_id
        self._next_id += batch_size
        return self.images[cur_id:cur_id+batch_size], self.labels[cur_id:cur_id+batch_size]

    def get(self):
        return self.images, self.labels

class Dataset_all(object):
    def __init__(self, train_csv, test_csv, dev_prob=0.1, label=0, one_hot_num=10, scale=256):
        test_data = np_load_csv(test_csv, add_dummy=True, dummy_col=label)
        train_data_orig = np_load_csv(train_csv)
        train_data, dev_data = train_test_split(train_data_orig, test_size=dev_prob)
        self.train      = Dataset(train_data, label=label, one_hot_num=one_hot_num, scale=scale)
        self.validation = Dataset(dev_data,   label=label, one_hot_num=one_hot_num, scale=scale)
        self.test       = Dataset(test_data,  label=label, one_hot_num=one_hot_num, scale=scale)



#%% here is the functions for defining neural network
def weight_bias(shape):
    W = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[shape[-1]]))
    return W, b

def fully_connect_layer(x, shape, keep_prob, actf='relu', pool='max2'): # shape [n_in, n_hidden, n_out]
    W1, b1 = weight_bias(shape[0:2])
    W2, b2 = weight_bias(shape[1:3])
    h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    h1_drop = tf.nn.dropout(h1, keep_prob)
    y = tf.nn.softmax(tf.matmul(h1_drop, W2) + b2)
    return y

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool(x, n=2):
    return tf.nn.max_pool(x, ksize=[1,n,n,1], strides=[1,n,n,1], padding='SAME')

def convolution_layer(X, shape): # shape = [filter_height, filter_width, in_channels, out_channels]
    W, b = weight_bias(shape)
    h_cv = tf.nn.relu(conv2d(X, W) + b)
    h_pl = max_pool(h_cv)
    return h_pl