from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf, math

WEIGHT_DECAY = 0.0001
BN = False
MOMENTUM = 0.9
EPSILON = 1e-4
DROPOUT = 0.8

def _activation_summary(x):
    tensor_name = x.op.name
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
    
def _variable_trunc_normal(name, shape, trainable=True):
    return tf.get_variable(name, shape,
            initializer=tf.truncated_normal_initializer(stddev=0.01),
            trainable=trainable)

def _variable_constant(name, shape, value, trainable=True):
    return tf.get_variable(name, shape,
            initializer=tf.constant_initializer(value),
            trainable=trainable)

def _batch_normalization(x, n_out, phase, momentum_list):
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
    mean, var = tf.cond(
        tf.Variable(phase=='train', trainable=False),
        update,
        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    x = tf.nn.batch_normalization(x, mean, var, \
        beta, gamma, EPSILON)
    return x



def _add_weight_decay(var, wd):
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('weight_losses', weight_decay)

def _conv(input, shape, strides, weight_decay, phase, name, alpha=0.1):
    """
    args:
        shape : [3, 3, in, out]
    """
    
    if BN:
        with tf.variable_scope(name) as scope:
            kernel = _variable_trunc_normal('weights', shape)
            conv = tf.nn.conv2d(input, kernel, strides, padding='SAME')
            # shape of conv : [batch_size, height, width, chanels]
    
            bn_conv = _batch_normalization(conv, shape[-1],
                    phase, [0, 1, 2])
            conv_ = tf.maximum(bn_conv, alpha*bn_conv, name=scope.name)
            if tf.get_variable_scope().reuse is False:
                _add_weight_decay(kernel, weight_decay)
                _activation_summary(conv_)
    else:
        with tf.variable_scope(name) as scope:
            kernel = _variable_trunc_normal('weights', shape)
            conv = tf.nn.conv2d(input,kernel,strides, padding='SAME')
            biases = _variable_constant('biases', shape[-1], value=0.01)
            bias = tf.nn.bias_add(conv, biases)
            conv_ = tf.maximum(bias, alpha*bias, name=scope.name)
            if tf.get_variable_scope().reuse is False:
                _add_weight_decay(kernel, weight_decay)
                _activation_summary(conv_)

    return conv_

def _fc(input, shape, weight_decay, phase, dropout, name, alpha=0.1, \
        last=False):
    if BN and not last:
        with tf.variable_scope(name) as scope:
            weights = _variable_trunc_normal('weights', shape)
            fc = tf.matmul(input, weights)
            fc = _batch_normalization(fc, shape[-1], phase, [0])
            fc = tf.maximum(fc, alpha*fc, name=scope.name)
            if tf.get_variable_scope().reuse is False:
                _add_weight_decay(weights, weight_decay)
                _activation_summary(fc)
    else:
         with tf.variable_scope(name) as scope:
            weights = _variable_trunc_normal('weights', shape)
            biases = _variable_constant('biases', shape[-1], value=0.0)
            fc = tf.matmul(input, weights) + biases
            if not last:
                fc = tf.maximum(fc, alpha*fc, name=scope.name)
            if tf.get_variable_scope().reuse is False:
                _add_weight_decay(weights, weight_decay)
                _activation_summary(fc)
    return fc

def ConvLayer(input, in_filter, out_filter, weight_decay, phase, dropout, layer_num):
    conv_strides = [1,1,1,1]
    pool_strides = [1,2,2,1]
    ksize = [1,2,2,1]

    _conv1 = _conv(input, [3,3,in_filter,out_filter], conv_strides,
                weight_decay, phase, name = 'conv%d1'%layer_num)
    _conv2 = _conv(_conv1, [3,3,out_filter,out_filter], conv_strides,
                weight_decay, phase, name = 'conv%d2'%layer_num)
    _pool = tf.nn.max_pool(_conv2, ksize, pool_strides, padding='VALID',
                name = 'pool%d'%layer_num)
    _drop = tf.nn.dropout(_pool, dropout)

    return _drop


def inference(images, batch_size, phase):
    weight_decay = WEIGHT_DECAY
    dropout = tf.cond(tf.Variable(phase=='train', trainable=False),
            lambda: tf.Variable(DROPOUT, trainable=False),
            lambda: tf.Variable(1.0, trainable=False))


    layer0 = ConvLayer(images, 3, 32, weight_decay, phase, dropout, 0)
    layer1 = ConvLayer(layer0, 32, 64, weight_decay, phase, dropout, 1)
    layer2 = ConvLayer(layer1, 64, 128, weight_decay, phase, dropout, 2)
    layer3 = ConvLayer(layer2, 128, 256, weight_decay, phase, dropout, 3)
    
    # fc layer
    dim = 1
    for d in layer3.get_shape()[1:].as_list():
        dim *= d
    nn_in = tf.reshape(layer3, [batch_size, dim])
    nn_in = tf.nn.dropout(nn_in, dropout)

    fc1 = _fc(nn_in, [dim, 600], weight_decay, phase, dropout,
            name = 'fc1')
    drop_f1 = tf.nn.dropout(fc1, dropout)
    fc2 = _fc(drop_f1, [600,600], weight_decay, phase, dropout,
        name = 'fc2')
    drop_f2 = tf.nn.dropout(fc2, dropout)
    fc3 = _fc(drop_f2, [600,10], False, phase, dropout,
            name='fc3', last=True)

    return fc3
