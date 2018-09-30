# ***
# ******************************************************************************
# * @filename: Create Neural Network with regularization
# * @author  : tjwx
# * @version :
# * @date    : 2018.09.26
# * @brief   : This file provides all the  main functions.
# * @reference:Google TensorFlow  practice in chapter 3.4
# ******************************************************************************
# *-*- coding: utf-8 -*-
# * @TODO : None
# * @Note : Network Structure
# *         input ；Tensor[n,2] , hidden layer : [10,10,10] output:Tensor [n,1]
# *
# ******************************************************************************
# *

import tensorflow as tf
from tensorflow.contrib import layers as ly

# define weights of one side NN(right side)
def get_weight(shape,lam):
	"""
	:param shape: the shape of generate weight variables
	:param lam: the parameter of regularization
	:return: a weight variable
	"""
	var = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
	#add the var to the collection
	tf.add_to_collection('losses',ly.l2_regularizer(lam)(var))
	return var


# define Neural Network structure
x = tf.placeholder(tf.float32,[None,2],'x_input')
y_= tf.placeholder(tf.float32,[None,1],'y_input')
batch_size = 8
#define the layers number and the nodes number of each layer(including input and output layer)
layers_dimension = [2,10,10,10,1]
n_layers = len(layers_dimension)
#define current layer input and node number
current_layer = x
current_dimension = layers_dimension[0]
# construct a full connection NN with 5 layers using for, namely
for i in range(1,n_layers):
	next_dimension = layers_dimension[i]
	# generate the weights and add their L2 regularization to losses collection
	weight = get_weight([current_dimension,next_dimension],0.001)
	bias = tf.Variable(tf.constant(0.1,shape=[next_dimension]))
	current_layer = tf.nn.relu(tf.matmul(current_layer,weight) + bias)
	current_dimension = next_dimension
y = current_layer
#calculate the loss on dataset
mse_loss = tf.reduce_mean(tf.square(y_ - y))
# add the mse_loss on dataset to the collection including the L2
# regularization of each layer weight
tf.add_to_collection('losses',mse_loss)
# add the all collection variables together to form the total loss
loss = tf.add_n(tf.get_collection('losses'))
