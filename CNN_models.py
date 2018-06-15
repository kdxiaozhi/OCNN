#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 12:24:35 2017

@author: zhaowenzhi
"""
import tensorflow as tf

def model_28_plain(X, w, w2, w3, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 24, 24, 32)
                        strides=[1, 1, 1, 1], padding='VALID'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 12, 12, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 10, 10, 64)
                        strides=[1, 1, 1, 1], padding='VALID'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 5, 5, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3 = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 1, 1, 128)
                        strides=[1, 1, 1, 1], padding='VALID'))
#    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
#                        strides=[1, 2, 2, 1], padding='SAME')
    l3shape = l3.get_shape().as_list()
    l3 = tf.reshape(l3, [-1, l3shape[1] * l3shape[2] * l3shape[3]])    # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    #l4 = tf.nn.relu(tf.matmul(l3, w4))
    #l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l3, w_o)
    return pyx