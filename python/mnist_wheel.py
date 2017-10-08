#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 19:18:53 2017

@author: yyang
"""

import foo
import dnn_wheel as wh

# here is the data reading from example
#from tensorflow.examples.tutorials.mnist import input_data
#mnist0 = input_data.read_data_sets("../MNIST/MNIST_data/", one_hot=True)

mnist = foo.Dataset_all('../mnist_data/train.csv', '../mnist_data/test.csv')
ii = 3010
tmp = mnist.train
foo.display_flatten_image(tmp.images[ii])
print tmp.labels[ii]


#%%
reload(foo)
reload(wh)

n_pixel = 784
n_label = 10
n_hidden = 50

keep_prob = 0.5
cfg = dict()
cfg['weight_sigma'] = 'he'
cfg['alpha'] = 1

m1 = wh.simple_nn_model([784, 300, 10], ['relu', 'dropout_softmax'], cfg=cfg)

for k in range(5000):
    bx, by = mnist.train.next_batch(150)
    cfg['keep_prob'] = keep_prob
    m1.train(bx, by)
    if (k+1) % 100 == 0:
        bx, by = mnist.validation.get()
        cfg['keep_prob'] = 1
        m1.evaluate(bx)
        pe, E = m1.check(by)
        print ('%.2f%%' % (100*pe)), E
    if (k+1) % 1000 == 0:
        cfg['alpha'] *= 0.3
        print 'gear shift to %f' % (cfg['alpha'])
