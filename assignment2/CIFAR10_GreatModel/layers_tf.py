import tensorflow as tf
import sys
sys.path.append('..')
from cs231n.data_utils import load_CIFAR10
import numpy as np


# load cifar10 dataset
def get_CIFAR10_data(num_training=45000, num_validation=5000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = '../cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test


# define batch normalization layer
def batch_norm_wrapper(inputs, is_training, decay = 0.999, epsilon=1e-6):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training is True:
        batch_mean, batch_var = tf.nn.moments(inputs, axes=[0,1,2])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)


def conv_norm_relu_pool(x, num_filters, filter_size, pool_size, training, init, reg, scope):
    with tf.variable_scope(scope):
        h1 = tf.layers.conv2d(x, num_filters, filter_size, padding='same', kernel_initializer=init, kernel_regularizer=reg, name='conv')
        h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=training, scope='bn')
        h3 = tf.nn.relu(h2, name='relu')
        h4 = tf.layers.max_pooling2d(h3, pool_size=pool_size, strides=pool_size)
        return h4


def dense_norm_relu(x, fc_size, training, init, reg, scope):
    with tf.variable_scope(scope):
        h1 = tf.layers.dense(x, fc_size, kernel_initializer=init, kernel_regularizer=reg, name='dense')
        h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=training, scope='bn')
        return tf.nn.relu(h2, name='relu')


def affine_norm_relu_dropout(x, fc_size, drop_rate, training, init, reg, scope):
    with tf.variable_scope(scope):
        h1 = tf.layers.dense(x, fc_size, kernel_initializer=init, kernel_regularizer=reg, name='dense')
        h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=training, scope='bn')
        h3 = tf.nn.relu(h2, name='relu')
        h4 = tf.layers.dropout(h3, rate=drop_rate, training=training, name='drop')
        return h4


def affine_relu_dropout(x, fc_size, drop_rate, training, init, reg, scope):
    with tf.variable_scope(scope):
        h1 = tf.layers.dense(x, fc_size, activation=tf.nn.relu, kernel_initializer=init, kernel_regularizer=reg, name='dense')
        h2 = tf.layers.dropout(h1, rate=drop_rate, training=training, name='drop')
        return h2
