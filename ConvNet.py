from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf, math

WEIGHT_DECAY = 0.0005
BN = False
MOMENTUM = 0.9
EPSILON = 1e-4

class ConvNet(object):

    def _activation_summary(self,x):
        tensor_name = x.op.name
        tf.histogram_summary(tensor_name + '/activations', x)
        tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
    
    def _variable_trunc_normal(self,name, shape, trainable=True):
        return tf.get_variable(name, shape,
            initializer=tf.truncated_normal_initializer(stddev=0.01),
            trainable=trainable)

    def _variable_constant(self,name, shape, value, trainable=True):
        return tf.get_variable(name, shape,
            initializer=tf.constant_initializer(value),
            trainable=trainable)

    def _batch_normalization(self, x, n_out, momentum_list):
        gamma = tf.get_variable('gamma', [n_out],
            initializer = tf.constant_initializer(1.0), trainable=True)
        beta = tf.get_variable('beta', [n_out],
            initializer = tf.constant_initializer(0.0), trainable=True)
    
        batch_mean, batch_var = tf.nn.moments(x, momentum_list, \
                    name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=MOMENTUM)
        def update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return (tf.identity(batch_mean), tf.identity(batch_var))
        mean, var = tf.cond(self.train_mode,
            update,
            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        x = tf.nn.batch_normalization(x, mean, var, \
            beta, gamma, EPSILON)
        return x

    def _add_weight_decay(self, var):
        if self.weight_decay:
            weight_decay = tf.mul(tf.nn.l2_loss(var), self.weight_decay,
                    name='weight_loss')
            tf.add_to_collection('weight_losses', weight_decay)

    def _conv(self, input, shape, strides, name, alpha=0.1):
        """
        args:
            shape : [3, 3, in, out]
        """
    
        if self.bn_mode:
            with tf.variable_scope(name) as scope:
                kernel = self._variable_trunc_normal('weights', shape)
                conv = tf.nn.conv2d(input, kernel, strides, padding='SAME')
                bn_conv = self._batch_normalization(conv, shape[-1], [0, 1, 2])
                conv_ = tf.maximum(bn_conv, alpha*bn_conv, name=scope.name)
                if tf.get_variable_scope().reuse is False:
                    self._add_weight_decay(kernel)
                    self._activation_summary(conv_)
        else:
            with tf.variable_scope(name) as scope:
                kernel = self._variable_trunc_normal('weights', shape)
                conv = tf.nn.conv2d(input,kernel,strides, padding='SAME')
                biases = self._variable_constant('biases', shape[-1], value=0.01)
                bias = tf.nn.bias_add(conv, biases)
                conv_ = tf.maximum(bias, alpha*bias, name=scope.name)
                if tf.get_variable_scope().reuse is False:
                    self._add_weight_decay(kernel)
                    self._activation_summary(conv_)

        return conv_

    def _fc(self, input, shape, name, alpha=0.1, last=False):
    
        if self.bn_mode and not last:
            with tf.variable_scope(name) as scope:
                weights = self._variable_trunc_normal('weights', shape)
                fc = tf.matmul(input, weights)
                fc = self._batch_normalization(fc, shape[-1], [0])
                fc = tf.maximum(fc, alpha*fc, name=scope.name)
                if tf.get_variable_scope().reuse is False:
                    self._add_weight_decay(weights)
                    self._activation_summary(fc)
        else:
            with tf.variable_scope(name) as scope:
                weights = self._variable_trunc_normal('weights', shape)
                biases = self._variable_constant('biases', shape[-1], value=0.0)
                fc = tf.matmul(input, weights) + biases
                if not last:
                    fc = tf.maximum(fc, alpha*fc, name=scope.name)
                if tf.get_variable_scope().reuse is False:
                    self._add_weight_decay(weights)
                    self._activation_summary(fc)
        return fc

    def ConvLayer(self, input, in_filter, out_filter, layer_num):
        conv_strides = [1,1,1,1]
        pool_strides = [1,2,2,1]
        ksize = [1,2,2,1]

        _conv1 = self._conv(input, [3,3,in_filter,out_filter], conv_strides,
                name = 'conv%d1'%layer_num)
        _conv2 = self._conv(_conv1, [3,3,out_filter,out_filter], conv_strides,
                name = 'conv%d2'%layer_num)
        _pool = tf.nn.max_pool(_conv2, ksize, pool_strides, padding='VALID',
                name = 'pool%d'%layer_num)
        _drop = tf.nn.dropout(_pool, self.dropout)

        return _drop

    def __init__(self, dropout = 0.5, hidden_dim = 1000, bn_mode = False):
        self.weight_decay = WEIGHT_DECAY
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.bn_mode = bn_mode

    def inference(self, images, batch_size, train_mode):
        self.train_mode = train_mode
        self.dropout = tf.cond(train_mode,
            lambda: tf.Variable(self.dropout, trainable=False),
            lambda: tf.Variable(1.0, trainable=False))


        layer0 = self.ConvLayer(images, 3, 64, 0)
        layer1 = self.ConvLayer(layer0, 64, 128, 1)
        layer2 = self.ConvLayer(layer1, 128, 256, 2)
        layer3 = self.ConvLayer(layer2, 256, 512, 3)
    
        # fc layer
        dim = 1
        for d in layer3.get_shape()[1:].as_list():
            dim *= d
        nn_in = tf.reshape(layer3, [batch_size, dim])
        nn_in = tf.nn.dropout(nn_in, self.dropout)

        fc1 = self._fc(nn_in, [dim, self.hidden_dim], name = 'fc1')
        drop_f1 = tf.nn.dropout(fc1, self.dropout)
        fc2 = self._fc(drop_f1, [self.hidden_dim, self.hidden_dim], name = 'fc2')
        drop_f2 = tf.nn.dropout(fc2, self.dropout)
        fc3 = self._fc(drop_f2, [self.hidden_dim,10], name='fc3', last=True)

        return fc3
