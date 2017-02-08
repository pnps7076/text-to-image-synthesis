#!/usr/bin/python
import numpy as np
import tensorflow as tf


def lrelu(x,leak=0.2):
    return tf.maximum(x,leak*x)

def conv2d(x,out_dim,k_h=5,k_w=5,strides=[1,2,2,1],padding='SAME',train=True,name='conv2d',act=True):
    input_shape = x.get_shape()
    with tf.variable_scope(name):
        w = tf.get_variable(name = 'filter',
                            dtype = tf.float32,
                            shape = [k_h,k_w,input_shape[-1],out_dim],
                            initializer = tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable(name = 'bias',
                            dtype = tf.float32,
                            shape = [out_dim],
                            initializer = tf.constant_initializer(0.0))
        conv = (tf.nn.bias_add(tf.nn.conv2d(x,w,strides = strides,padding = padding),b))
        if train:
            conv = tf.contrib.layers.batch_norm(conv,is_training = train)
        if act:
            conv = lrelu(conv)
    return conv

def conv2d_transpose(x,output_shape,k_h=5,k_w=5,strides=[1,2,2,1],padding='SAME',train=True,name='upconv2d',act=True):
    input_shape = x.get_shape()
    with tf.variable_scope(name):
        w = tf.get_variable(name = 'filter',
                            dtype = tf.float32,
                            shape = [k_h,k_w,output_shape[-1],input_shape[-1]],
                            initializer = tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable(name = 'bias',
                            dtype = tf.float32,
                            shape = [output_shape[-1]],
                            initializer = tf.constant_initializer(0.0))
        conv = (tf.nn.bias_add(tf.nn.conv2d_transpose(x,w,
                                                        output_shape = output_shape,
                                                        strides = strides,
                                                        padding = padding),b))
        if train:
            conv = tf.contrib.layers.batch_norm(conv,is_training = train)
        if act:
            conv = lrelu(conv)
    return conv

def Linear(x,out_shape,train=True,name='linear',act=True):
    input_shape = x.get_shape()
    with tf.variable_scope(name):
        w = tf.get_variable(name = 'weights',
                            dtype = tf.float32,
                            shape = [input_shape[-1],out_shape],
                            initializer=tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable(name = 'bias',
                            dtype = tf.float32,
                            shape = [out_shape],
                            initializer = tf.constant_initializer(0.0))
        out = tf.nn.bias_add(tf.matmul(x,w),b)
        if train:
            out = tf.contrib.layers.batch_norm(out,is_training = train)
        if act:
            out = lrelu(out)
    return out
