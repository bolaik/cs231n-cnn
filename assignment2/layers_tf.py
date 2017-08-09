import tensorflow as tf

# define batch normalization layer
def batch_norm_wrapper(inputs, is_training, decay = 0.999, epsilon=1e-6, axes=[0,1,2]):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training is True:
        batch_mean, batch_var = tf.nn.moments(inputs, axes=axes)
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)


def conv_relu_pool(x, num_filters, filter_size, padding, pool_size, pool_strides, init):
    layer_conv = tf.layers.conv2d(inputs=x, filters=num_filters, kernel_size=filter_size, activation=tf.nn.relu,
                                  padding=padding, kernel_initializer=init)
    layer_pool = tf.layers.max_pooling2d(inputs=layer_conv, pool_size=pool_size, strides=pool_strides)
    return layer_pool


def conv_norm_relu_pool(x, num_filters, filter_size, padding, pool_size, pool_strides, training, init):
    layer_conv = tf.layers.conv2d(inputs=x, filters=num_filters, kernel_size=filter_size,
                                  padding=padding, kernel_initializer=init)
    layer_bn = batch_norm_wrapper(inputs=layer_conv, is_training=training)
    layer_relu = tf.nn.relu(layer_bn)
    layer_pool = tf.layers.max_pooling2d(inputs=layer_relu, pool_size=pool_size, strides=pool_strides)
    return layer_pool


def affine_norm_relu(x, fc_size, training, init, axes):
    layer_affine = tf.layers.dense(inputs=x, units=fc_size, kernel_initializer=init)
    layer_bn = batch_norm_wrapper(inputs=layer_affine, is_training=training, axes=axes)
    layer_relu = tf.nn.relu(layer_bn)
    return layer_relu


def affine_norm_relu_dropout(x, fc_size, drop_rate, training, init, axes):
    layer_affine = tf.layers.dense(inputs=x, units=fc_size, kernel_initializer=init)
    layer_bn = batch_norm_wrapper(inputs=layer_affine, is_training=training, axes=axes)
    layer_relu = tf.nn.relu(layer_bn)
    layer_drop = tf.layers.dropout(inputs=layer_relu, rate=drop_rate, training=training)
    return layer_drop


def affine_relu_dropout(x, fc_size, drop_rate, training, init):
    layer_dense = tf.layers.dense(inputs=x, units=fc_size, activation=tf.nn.relu,
                                  kernel_initializer=init)
    layer_drop = tf.layers.dropout(inputs=layer_dense, rate=drop_rate, training=training)
    return layer_drop