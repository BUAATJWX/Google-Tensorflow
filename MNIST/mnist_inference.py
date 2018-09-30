# ***
# ******************************************************************************
# * @function: define the NN structure and arguments , the process of forward
# *            propagation
# * @author  : tjwx
# * @version :
# * @date    : 2018.09.27
# * @brief   : This file provides all the functions.
# * @reference:Google TensorFlow  practice in chapter 5.5
# ******************************************************************************
# *-*- coding: utf-8 -*-
# * @TODO : None
# * @Note : Network Structure
# *         input ；Tensor[n,748] , hidden layer : [500] output:Tensor [n,10]
# *
# ******************************************************************************
# *

import tensorflow as tf

tf.set_random_seed(1)
# define the structure of NN
input_node = 784
output_node = 10
layer1_node = 500

# define function got weight
def get_weight_variable(shape, regularizer):
	"""
	:func  generate the weights and corresponding regularization  which will be added
			to the collection 'losses'
	:param shape: the shape of weights to be created
	:param regularizer: the regularization rate
	:return: the weights
	"""
	weights = tf.get_variable('weights', shape=shape,
							  initializer=tf.truncated_normal_initializer(stddev=0.1))
	if regularizer is not None:
		tf.add_to_collection('losses', regularizer(weights))
	return weights


# define the forward propagation and arguments
def inference(input_tensor, avg_class, regular,reuse=False):
	"""
	:func  create the hidden layer node using given parameter and Relu
			activate function and calculation the forward propagation
	:param input_tensor: input
	:param regular: regularization rate
	:param reuse:  use the weights existed when reuse is true
	:param avg_class: use the Moving average method
	:return: the forward propagation calculation of output
	"""
	# define the first layer
	with tf.variable_scope('layer1', reuse=reuse):
		weights = get_weight_variable([input_node, layer1_node], regular)
		biases = tf.get_variable('biases',
								 shape=[layer1_node], initializer=tf.constant_initializer(1.0))
		if avg_class == None:
			layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
		else:
			layer1 = tf.nn.relu(
				tf.matmul(input_tensor, avg_class.average(weights)) + avg_class.average(biases))

	# define the second layer
	with tf.variable_scope('layer2', reuse=reuse):
		weights = get_weight_variable([layer1_node, output_node], regular)
		biases = tf.get_variable('biases',
								 shape=[output_node], initializer=tf.constant_initializer(1.0))
		if avg_class == None:
			layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)
		else:
			layer2 = tf.nn.relu(
				tf.matmul(layer1, avg_class.average(weights)) + avg_class.average(biases))
	return layer2



