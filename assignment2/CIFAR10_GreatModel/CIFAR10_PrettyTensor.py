import numpy as np
import os
import time
from datetime import timedelta
from cs231n.data_utils import load_CIFAR10
from layers_tf import *
import prettytensor as pt

def get_CIFAR10_data(num_training=45000, num_validation=5000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
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


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape:\t\t', X_train.shape)
print('Train labels shape:\t\t', y_train.shape)
print('Validation data shape:\t\t', X_val.shape)
print('Validation labels shape:\t', y_val.shape)
print('Test data shape:\t\t', X_test.shape)
print('Test labels shape:\t\t', y_test.shape)

# data dimension
img_size = 32
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 3
num_classes = 10
print_every = 1000

# clear old variables
tf.reset_default_graph()

x_image = tf.placeholder(tf.float32, [None, 32, 32, 3])
y_true_cls = tf.placeholder(tf.int64, [None])

def main_network(images, training):
    # Wrap the input images as a Pretty Tensor object.
    x_pretty = pt.wrap(images)

    # Pretty Tensor uses special numbers to distinguish between
    # the training and testing phases.
    if training:
        phase = pt.Phase.train
    else:
        phase = pt.Phase.infer

    # Create the convolutional neural network using Pretty Tensor.
    # It is very similar to the previous tutorials, except
    # the use of so-called batch-normalization in the first layer.
    with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
        y_pred, loss = x_pretty.\
            conv2d(kernel=3, depth=16, name='layer_conv1', batch_normalize=True).\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=3, depth=32, name='layer_conv2', batch_normalize=True).\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=3, depth=64, name='layer_conv3', batch_normalize=True).\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=3, depth=128,name='layer_conv4', batch_normalize=True).\
            max_pool(kernel=2, stride=2).\
            flatten().\
            fully_connected(size=256, name='layer_fc1').\
            fully_connected(size=256, name='layer_fc2').\
            softmax_classifier(num_classes=num_classes, labels=tf.one_hot(y_true_cls, 10))

    return y_pred, loss

# loss and optimizer
_, loss = main_network(x_image, training=True)
global_step = tf.Variable(initial_value=0, trainable=False)
learning_rate_init = 1e-3
decay_steps = 1000
learning_rate = tf.train.exponential_decay(learning_rate_init, global_step, decay_steps, 0.95, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# inference
y_pred, _ = main_network(x_image, training=False)
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# build saver to save the variables of neural network
saver = tf.train.Saver()
save_dir = 'checkpoints/cifar10'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation')


# create tensorflow session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# custom function for optimization
train_batch_size = 64

# the classification accuracy for the validation set will be calculated for every `print_every` iterations
# the optimization will stop if the validation accuracy has not been improved in `require_improvement` iterations
# variables to keep track of best model
best_val_acc = 0
last_improvement = 0
require_improvement = 10000

def random_batch(images, labels):
    num_images = len(images)
    idx = np.random.choice(num_images, size=train_batch_size, replace=False)
    x_batch = images[idx,:,:,:]
    y_batch = labels[idx]
    return x_batch, y_batch

def optimize(num_iterations):
    global best_val_acc
    global last_improvement

    start_time = time.time()
    for i in range(num_iterations):
        # get batch for training
        x_batch, y_true_batch = random_batch(X_train, y_train)
        feed_dict_train = {x_image: x_batch, y_true_cls: y_true_batch}
        _, lr = sess.run([optimizer, learning_rate], feed_dict=feed_dict_train)
        # print status
        if i % print_every == 0 or i == num_iterations - 1:
            # train accuracy
            acc_train = sess.run(accuracy, feed_dict=feed_dict_train)
            # validation accuracy
            _, correct_val = infer_cls(images=X_val, labels=y_val)
            acc_val = np.mean(correct_val)
            # check if improve over the saved best
            if acc_val > best_val_acc:
                best_val_acc = acc_val
                last_improvement = i
                # save tensorflow variables to file
                saver.save(sess=sess, save_path=save_path)
                improved_str = '*'
            else:
                improved_str = ''
            # status message for printing.
            msg = "Iter: {0:>6}, Learning Rate: {1:>.4E}, Train-Batch Accuracy: {2:>6.1%}, Validation Accuracy: {3:>6.1%} {4}"
            print(msg.format(i + 1, lr, acc_train, acc_val, improved_str))
        if i - last_improvement > require_improvement:
            print("No improvement found in a while, stopping optimization.")
            break

    end_time = time.time()
    time_delta = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_delta)))))


# calculate prediction performance
infer_batch_size = 256
def infer_cls(images, labels):
    num_images = len(images)
    cls_pred = np.zeros(num_images, dtype=np.int)
    i = 0
    while i < num_images:
        j = min(i+infer_batch_size, num_images)
        feed_dict = {x_image: images[i:j, :], y_true_cls: labels[i:j]}
        cls_pred[i:j] = sess.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    correct = (labels == cls_pred)
    return cls_pred, correct


def print_test_accuracy():
    cls_pred_test, correct_test = \
        infer_cls(images=X_test, labels=y_test)
    acc_test = np.mean(correct_test)
    num_correct_test = np.sum(correct_test)
    num_test_images = len(X_test)
    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc_test, num_correct_test, num_test_images))


# check performance after 1000 iteractions
optimize(num_iterations=100000)
print_test_accuracy()

# restore best variables
sess.run(tf.global_variables_initializer())
saver.restore(sess=sess, save_path=save_path)
print_test_accuracy()
