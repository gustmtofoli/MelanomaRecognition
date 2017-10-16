import cv2
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import math
from Preprocessing import load_images
import InfoMessages as Info
import tkinter as tk
from tkinter import filedialog
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.utils import np_utils
import tensorflow.contrib.slim as slim

# TODO separar front do back
# TODO verificar quais imagens foram classificadas corretamente
# TODO tempo de execução
# TODO adicionar base de dados no projeto
# TODO adicionar projeto no repositório
# TODO pythonQt
# TODO cnn_melanoma_teste2.py com augmentation, loss, 3 poolings, etc...


def _validationList(list):
    if len(list) == 0:
        return Info.errorMessage()


def _validationPath(path):
    if path is None:
        return Info.errorMessage()


def _validationType(type):
    if type == "" or type == None:
        return Info.errorMessage()


def chooseFolder(type):
    _validationType(type)
    tk.Tk().withdraw()
    path = filedialog.askdirectory()
    _validationPath(path)
    Info.infoFolder(path, type)
    return path


def loadImages(path, type):
    _validationType(type)
    image_List = load_images(path)
    _validationList(image_List)
    image_List = np.array(image_List)
    image_List.shape
    image_List = np.reshape(image_List,[-1, int(np.prod(image_List.shape[1:]))])
    image_List.shape
    Info.infoImages(image_List)
    return image_List


def center_normalize(x):
    return (x-np.mean(x))/np.std(x)


path_Train = chooseFolder("TRAIN")
path_Test = chooseFolder("TEST")
X_Train = loadImages(path_Train, "TRAIN")

X_Test = loadImages(path_Test, "TEST")
path_ground_truth_train = "/home/gustavo/Documentos/DATA_BASE/ISIC-2017_Training_Part3_GroundTruth.csv"
data_Train = pd.read_csv(path_ground_truth_train)
Info.infoGroundTruth(path_ground_truth_train, "TRAIN")
path_ground_truth_test = "/home/gustavo/Documentos/DATA_BASE/ISIC-2017_Test_v2_Part3_GroundTruth.csv";
data_Test = pd.read_csv(path_ground_truth_test)
Info.infoGroundTruth(path_ground_truth_test, "TEST")
data_Train = data_Train.iloc[0:2000,1]
print(data_Train)

data_Test = data_Test.iloc[0:600,1]
x_Train = data_Train
x_Test = data_Test
x_Train = np.array(x_Train)

x_Test = np.array(x_Test)
y_Train = data_Train
y_Test = data_Test
y_Train = np.array(y_Train)

y_Test = np.array(y_Test)
y_Train = np_utils.to_categorical(data_Train)


y_Test = np_utils.to_categorical(data_Test)


X_Train = center_normalize(X_Train)
X_Test = center_normalize(X_Test)
print("> [OK] Images are normalized")

print("> [STARTED] Convolution Neural Network")

tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, 4096*3],name="x-in")
true_y = tf.placeholder(tf.float32, [None, 2],name="y-in")
keep_prob = tf.placeholder("float")

x_image = tf.reshape(x,[-1,64,64,3])
hidden_1 = slim.conv2d(x_image, 32,[2,2])
pool_1 = slim.max_pool2d(hidden_1,[2,2])
hidden_2 = slim.conv2d(pool_1,8,[2,2])
pool_2 = slim.max_pool2d(hidden_2,[2,2])
hidden_3_1 = slim.conv2d(pool_2, 32, [2,2])
pool_3 = slim.max_pool2d(hidden_3_1, [2, 2])
hidden_3_2 = slim.conv2d(pool_3, 64, [2,2])
hidden_3 = slim.dropout(hidden_3_2, keep_prob)
out_y = slim.fully_connected(slim.flatten(hidden_3),2,activation_fn=tf.nn.softmax)


cross_entropy = -tf.reduce_sum(true_y*tf.log(out_y))
correct_prediction = tf.equal(tf.argmax(out_y,1), tf.argmax(true_y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

batchSize = 100
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(1001):
    train_idx=np.random.randint(X_Train.shape[0],size=100)
    batch_x_Train = X_Train[train_idx,:]
    batch_y_Train = y_Train[train_idx]
    sess.run(train_step, feed_dict={x:batch_x_Train,true_y:batch_y_Train, keep_prob:0.5})
    if i % 100 == 0 and i != 0:
        trainAccuracy = sess.run(accuracy, feed_dict={x:batch_x_Train,true_y:batch_y_Train, keep_prob:1.0})
        Info.infoTrainingAccuracy(i, trainAccuracy)

train_idx = np.random.randint(X_Test.shape[0],size=100)
batch_x_Test = X_Test[train_idx,:]
batch_y_Test = y_Test[train_idx]
testAccuracy = sess.run(accuracy, feed_dict={x:batch_x_Test,true_y:batch_y_Test, keep_prob:1.0})
Info.infoTestAccuracy(testAccuracy)