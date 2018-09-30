# ***
# ******************************************************************************
# * @filename: the test program of google TensorFlow  practice in chapter 3.4
# * @author  : tjwx
# * @version :
# * @date    : 2018.09.20
# * @brief   : This file provides all the  main functions.
# * @reference:
# ******************************************************************************
# *-*- coding: utf-8 -*-
# * @TODO : None
# * @Note : Network Structure
# *         input ；Tensor[n,2] , hidden layer : [3] output:Tensor [n,1]

import tensorflow as tf
import keras
import sys
from numpy.random import RandomState

#define the size of train batch
batch_size = 8

#define the weight of neural network
w1 = tf.Variable(tf.random_normal([2,3], stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1,seed=1))

#define input and output
x = tf.placeholder(tf.float32,shape=(None, 2),name='x_input')
y_ = tf.placeholder(tf.float32,shape=(None, 1),name='y_input')

#define forward propagation of NN
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#define backward propagation algorithm and lost function of NN
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#generate the data_set
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)

#determine the label of X
Y = [[int(x1 + x2 < 1)] for (x1,x2) in X]

# create a session to run the program in default graph
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	step = 50000
	for i in range(step):
		#select batch samples to train
		start = (i * batch_size )% dataset_size
		end = min(start+batch_size , dataset_size)
		sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
		if i % 1000 == 0 :
			total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
			print("training step: %d, cross entropy on all data：%g"% ( i, total_cross_entropy))
	print(sess.run(w1))
	print(sess.run(w2))
