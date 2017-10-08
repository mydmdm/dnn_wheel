#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 16:39:16 2017

@author: yyang
"""
from __future__ import division
import numpy as np


#%% activation functions
def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    g = y * (1 - y)
    return y, g


def softmax(x):
    s = np.exp(x)
    n = np.sum(s, axis=1)
    y = (s.T / n).T
    g = np.ones_like(y)
    return y, g


def relu(x):
    y = np.copy(x)
    g = np.ones_like(x)
    k = (x <= 0).nonzero()
    y[k] = 0
    g[k] = 0
    return y, g


def softmax_cross_entropy(y, y_):
    return - np.mean(np.sum(y_ * np.log(y), axis=1))


def sigmoid_cross_entropy(y, y_):
    return - np.mean(np.sum(y_ * np.log(y) + (1-y_) * np.log(1-y), axis=1))


def error_prob(y, y_):
    return np.sum(np.argmax(y, axis=1) != np.argmax(y_, axis=1)) / y.shape[0]


class activation_layer(object):
    def __init__(self, a_type):
        if 'sigmoid' in a_type:
            self.afunc = sigmoid
            self.lfunc = sigmoid_cross_entropy
        elif 'softmax' in a_type:
            self.afunc = softmax
            self.lfunc = softmax_cross_entropy
        elif 'relu' in a_type:
            self.afunc = relu
        else:
            print 'not supported %s' % (a_type)

    def forward(self, x_):
        x = np.copy(x_)
        y, self.grad = self.afunc(x)
        return y

    def backward(self, x, gy):
        gx = self.grad * gy
        return gx


#%% fully connected layer
def weights_init(shape, cfg=dict()):
    if not ('weight_sigma' in cfg):
        cfg['weight_sigma'] = 'he'
    w = np.random.randn(*shape)
    if cfg['weight_sigma'] == 'he':
        w *= np.sqrt(2.0 / w.shape[0])
    elif cfg['weight_sigma'] == 'xavier':
        w *= np.sqrt(1.0 / w.shape[0])
    else:
        w *= cfg['weight_sigma']
    return w


def bias_init(shape, cfg=dict()):
    b = np.zeros(shape)
    return b


class dropout_layer(object):
    def __init__(self, cfg=dict()):
        self.cfg = cfg

    def forward(self, x):
        return self.dropout_vector(x, ff=True)

    def backward(self, x, gy):
        return self.dropout_vector(gy, ff=False)

    def dropout_vector(self, x_, ff=True):
        if self.cfg['keep_prob'] == 1:
            return x_
        x = np.copy(x_)
        if ff:
            self.dropmask = np.random.rand(*x.shape) < self.cfg['keep_prob']
        x *= self.dropmask.astype('int') / self.cfg['keep_prob']
        return x


class neural_layer(object):
    def __init__(self, w_shape, b_shape, dropout=False, cfg=dict()):
        self.w = weights_init(w_shape, cfg=cfg)
        self.b = bias_init(b_shape, cfg=cfg)
        self.dropout_en = dropout
        self.cfg = cfg

    def forward(self, x):
        y = np.dot(x, self.w) + self.b
        return y

    def backward(self, x, gy):
        gx = np.dot(gy, self.w.T)
        self.optimize_basic_sgd(x, gy)
        return gx

    def optimize_basic_sgd(self, x, gy):
        n = x.shape[0]
        gw = np.dot(x.T, gy) / n
        gb = np.sum(gy, axis=0) / n
        self.w -= self.cfg['alpha'] * gw
        self.b -= self.cfg['alpha'] * gb


#%% simple neural network model
class simple_nn_model(object):
    def __init__(self, n_size, a_type, cfg=dict()):
        self.cfg = cfg
        self.layers = list()
        assert len(n_size) == len(a_type) + 1, 'layer configuration wrong'
        for k in range(len(a_type)):
            self.add_linear_activate_layer(n_size[k], n_size[k+1], a_type[k], cfg=cfg)

    def add_linear_activate_layer(self, n_in, n_out, a_type, cfg=dict()):
        if 'dropout' in a_type:
            self.layers.append(dropout_layer(cfg=cfg))
        self.layers.append(neural_layer([n_in, n_out], [1, n_out], cfg=cfg))
        self.layers.append(activation_layer(a_type))

    def evaluate(self, x):
        self.data = [x]
        for k in range(len(self.layers)):
            self.data.append(self.layers[k].forward(self.data[k]))

    def check(self, y_):
        E = self.layers[-1].lfunc(self.data[-1], y_)
        pe = error_prob(self.data[-1], y_)
        return pe, E

    def train(self, x, y_):
        self.evaluate(x)
        n = len(self.layers)
        self.grad = [[] for k in range(n)]
        self.grad[-1] = self.data[-1] - y_
        for k in range(n-2, -1, -1):
            self.grad[k] = self.layers[k].backward(self.data[k], self.grad[k+1])



