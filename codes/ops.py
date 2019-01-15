import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

class batch_norm(object):
            # h1 = lrelu(tf.contrib.layers.batch_norm(conv2d(h0, self.df_dim*2, name='d_h1_conv'),decay=0.9,updates_collections=None,epsilon=0.00001,scale=True,scope="d_h1_conv"))
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, scope=self.name)

def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                              (1. - targets) * tf.log(1. - preds + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

#def conv2d(input_, output_dim,
#           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
#           name="conv2d"):
#    with tf.variable_scope(name):
#        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
#                            initializer=tf.truncated_normal_initializer(stddev=stddev))
#        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
#
#        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
#        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
#
#        return conv

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d", weight_params=None, bias_params=None):
    with tf.variable_scope(name):
        if weight_params is not None:
            w = tf.Variable(weight_params, name='w',
                            dtype=tf.float32)
            biases = tf.Variable(bias_params, name='biases',
                                 dtype=tf.float32)

            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
            return conv
        else:
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))

            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
            return conv


def conv2dlayer(input_, output_dim, padding='SAME',    
                k_h=5, k_w=5, d_h=2, d_w=2,
                trainable=True, name="conv2d"):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(inputs=input_,
                                filters=output_dim,
                                kernel_size=(k_h, k_w),
                                strides=(d_h, d_w),
                                padding=padding,
                                trainable=trainable
                                )
        return conv


def deconv2d(input_, output_shape, padding='SAME',
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False, weight_params=None, bias_params=None):
    with tf.variable_scope(name):
        if weight_params is not None:

            w = tf.Variable(weight_params, name='w',
                            dtype=tf.float32)
            biases = tf.Variable(bias_params, name='biases',
                                 dtype=tf.float32)
            try:
                deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, padding=padding,
                                    strides=[1, d_h, d_w, 1])

            # Support for verisons of TensorFlow before 0.7.0
            except AttributeError:
                deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        else:
            # filter : [height, width, output_channels, in_channels]
            w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                initializer=tf.random_normal_initializer(stddev=stddev))

            try:
                deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

            # Support for verisons of TensorFlow before 0.7.0
            except AttributeError:
                deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

            biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def deconv2dlayer(input_, output_shape, padding='SAME',
                  k_h=5, k_w=5, d_h=2, d_w=2,
                  trainable=True, name="deconv2d"):
    with tf.variable_scope(name):
        deconv = tf.layers.conv2d_transpose(inputs=input_,
                                            filters=output_shape,
                                            kernel_size=(k_h, k_w),
                                            strides=(d_h,d_w),
                                            padding=padding,
                                            trainable=trainable)
        
        return deconv


def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

# def mlp(input, n_features, n_classes, n_hidden, dropout_keep_prob, name="mlp"):

#     with tf.variable_scope(name):
#                 # Store layers weight & bias MLP
#         weights = {
#             'c': tf.Variable(tf.random_normal([n_features, n_hidden]), name='c_w1', dtype=tf.float32),
#             'out': tf.Variable(tf.random_normal([n_hidden, n_classes]), name='c_w_out', dtype=tf.float32)
#         }
#         biases = {
#             'b': tf.Variable(tf.random_normal([n_hidden]), name='c_b1'),
#             'out': tf.Variable(tf.random_normal([n_classes]), name='c_b_out')
#         }
#         # tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])), dropout_keep_prob)
#         # Hidden fully connected layer
#         layer_1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(input, weights['c']), biases['b'])), dropout_keep_prob)
#         # Output fully connected layer with a neuron for each class
#         out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
#         return out_layer


#def deconv2d(input_, output_shape,
#             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
#             name="deconv2d", with_w=False):
#    with tf.variable_scope(name):
#        # filter : [height, width, output_channels, in_channels]
#        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
#                            initializer=tf.random_normal_initializer(stddev=stddev))
#
#        try:
#            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
#                                strides=[1, d_h, d_w, 1])
#
#        # Support for verisons of TensorFlow before 0.7.0
#        except AttributeError:
#            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
#                                strides=[1, d_h, d_w, 1])
#
#        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
#        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
#
#        if with_w:
#            return deconv, w, biases
#        else:
#            return deconv
