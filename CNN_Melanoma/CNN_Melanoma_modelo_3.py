from unittest.case import _BaseTestCaseContext

import numpy as np
from skimage.io import imread_collection
import tensorflow as tf
import pandas as pd
from Preprocessing import load_images
import tkinter as tk
from tkinter import filedialog
import math
import random
import tensorflow.contrib.slim as slim


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')


x = tf.placeholder(tf.float32, [None, 3, 10000*3])

y_ = tf.placeholder(tf.float32, [None, 2])

keep_prob = tf.placeholder("float")

#
# x_image = tf.reshape(x, [-1, 100, 100, 3])
#
# W_conv1 = weight_variable([5, 5, 3, 32])
#
# b_conv1 = bias_variable([32])
#
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#
# h_pool1 = max_pool_2x2(h_conv1)
#
# W_conv2 = weight_variable([5, 5, 32, 64])
#
# b_conv2 = bias_variable([64])
#
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#
# h_pool2 = max_pool_2x2(h_conv2)
#
# W_fc1 = weight_variable([40000, 1024])
#
# b_fc1 = bias_variable([1024])
#
# h_pool2_flat = tf.reshape(h_pool2, [-1, 40000])
#
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
# keep_prob = tf.placeholder(tf.float32)
#
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
# W_fc2 = weight_variable([1024, 2])
#
# b_fc2 = bias_variable([2])
#
# y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

x_image = tf.reshape(x,[-1,100,100,3])
hidden_1 = slim.conv2d(x_image, 16,[3,3])
pool_1 = slim.max_pool2d(hidden_1,[2,2])
Activation_layer_1 = tf.nn.relu(pool_1)
hidden_2 = slim.conv2d(Activation_layer_1,16,[3,3])
pool_2 = slim.max_pool2d(hidden_2,[2,2])
Activation_layer_2 = tf.nn.relu(pool_2)
hidden_3_1 = slim.conv2d(Activation_layer_2, 64, [3,3])
pool_3 = slim.max_pool2d(hidden_3_1, [2, 2])
Activation_layer_3 = tf.nn.relu(pool_3)

hidden_3 = slim.dropout(Activation_layer_3, keep_prob)
y_conv = slim.fully_connected(slim.flatten(hidden_3),2,activation_fn=tf.nn.softmax)


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_conv, labels = y_))

train_step = tf.train.AdamOptimizer(0.00001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))





tk.Tk().withdraw()
dir = filedialog.askdirectory()
print(dir)

data = load_images(dir)
data = np.asarray(data)
data.shape

labels = pd.read_csv("/home/gustavo/Documentos/DATA_BASE/1000_ISIC-2017_Training_Part3_GroundTruth.csv")
labels = labels.iloc[0:1000, 1]
labels = np.asarray(labels)

batch_size = 50
epochs = 100
percent = 0.9

data_size = len(data)
idx = np.arange(data_size)
random.shuffle(idx)
data = data[idx]
labels = labels[idx]

train = (data[0:np.int(data_size*percent):, :], labels[0:np.int(data_size*percent):])
test = (data[0:np.int(data_size*(1 - percent)):, :], labels[0:np.int(data_size*(1 - percent)):])
train_size = len(train[0])

sess = tf.InteractiveSession()
tf.initialize_all_variables().run()

for n in range(epochs):
    for i in range(int(np.ceil(train_size / batch_size))):
        if (i * batch_size + batch_size <= train_size):
            batch = (train[0][i * batch_size : i * batch_size + batch_size],
                     train[1][i * batch_size : i * batch_size + batch_size])
        else:
            batch = (train[0][i*batch_size:],
                     train[1][i*batch_size:])

    train_accuracy = train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})

    if (n%5 == 0):
        print("Epoca %d, acuracia do treinamento = %g" %(n, train_accuracy))
    #                    y_: batch_y_Train, keep_prob: 0.5}
    # feed_dict_train = {x: batch_x_Train,
    #
    # batch_y_Train = labels[train_idx]
    # batch_x_Train = data[train_idx, :]
    # train_idx = np.random.randint(data.shape[0], size=100)

# global total_iterations
# for i in range(total_iterations, total_iterations + num_iterations):





