import os
import time
from datetime import timedelta
from layers_tf import *

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

# configuration of cnn architecture
filter_size1, num_filters1 = 3, 16
filter_size2, num_filters2 = 3, 32
filter_size3, num_filters3 = 3, 64
filter_size4, num_filters4 = 3, 128
fc_size1, fc_size2 = 256, 256
print_every = 1000

# clear old variables
tf.reset_default_graph()

x_image = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x_image')
y_true_cls = tf.placeholder(tf.int64, [None], name='y_true_cls')
phase = tf.placeholder(tf.bool, name='phase')

# add initializer
init = tf.contrib.layers.xavier_initializer()

# add regularizer
reg = tf.contrib.layers.l2_regularizer(scale=0.05)

# add layers
conv1 = conv_norm_relu_pool(x_image, num_filters1, filter_size1, 2, phase, init, reg, 'conv1')
conv2 = conv_norm_relu_pool(conv1,   num_filters2, filter_size2, 2, phase, init, reg, 'conv2')
conv3 = conv_norm_relu_pool(conv2,   num_filters3, filter_size3, 2, phase, init, reg, 'conv3')
conv4 = conv_norm_relu_pool(conv3,   num_filters4, filter_size4, 2, phase, init, reg, 'conv4')
flat = tf.contrib.layers.flatten(conv4, scope='flat')
fc1 = dense_norm_relu(flat, fc_size1, phase, init, reg, 'fc1')
fc2 = dense_norm_relu(fc1,  fc_size2, phase, init, reg, 'fc2')
logits = tf.layers.dense(fc2, num_classes, kernel_initializer=init, kernel_regularizer=reg, name='logits')

with tf.name_scope('loss'):
    # predicted classes
    y_pred = tf.nn.softmax(logits)  # one-hot encoding
    y_pred_cls = tf.argmax(y_pred, dimension=1) # as class number

    # cost function to be optimized
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(y_true_cls, 10))
    base_loss = tf.reduce_mean(cross_entropy)
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([base_loss] + reg_loss)

with tf.name_scope('accuracy'):
    # performance measure
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# implement learning_rate decay
global_step = tf.Variable(initial_value=0, trainable=False)
learning_rate_init = 1e-3
decay_steps = 1000
learning_rate = tf.train.exponential_decay(learning_rate_init, global_step, decay_steps, 0.95, staircase=True)

# batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

'''
# plot computational graphics
from utils import show_graph
show_graph(tf.get_default_graph().as_graph_def())

# print all global variables
print([v.name for v in tf.global_variables()])
'''

# build saver to save the variables of neural network
saver = tf.train.Saver()
save_dir = 'checkpoints/cifar10'
if not os.path.exists(save_dir): os.makedirs(save_dir)
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
        feed_dict_train = {x_image: x_batch, y_true_cls: y_true_batch, phase: 1}
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
        feed_dict = {x_image: images[i:j, :], y_true_cls: labels[i:j], phase: 1}
        cls_pred[i:j] = sess.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    correct = (labels == cls_pred)
    return cls_pred, correct


def print_test_accuracy():
    _, correct_test = infer_cls(images=X_test, labels=y_test)
    acc_test = np.mean(correct_test)
    num_correct_test = np.sum(correct_test)
    num_test_images = len(X_test)
    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc_test, num_correct_test, num_test_images))


# check performance after 1000 iteractions
optimize(num_iterations=1)
print_test_accuracy()

# restore best variables
sess.run(tf.global_variables_initializer())
saver.restore(sess=sess, save_path=save_path)
print_test_accuracy()
