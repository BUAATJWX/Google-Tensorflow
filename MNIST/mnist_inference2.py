# ***
# ******************************************************************************
# * @function: define the CNN structure and arguments , the process of forward
# *            propagation
# * @author  : tjwx
# * @version :
# * @date    : 2018.09.27
# * @brief   : This file provides all the functions.
# * @reference:Google TensorFlow  practice in chapter 5.5
# ******************************************************************************
# *-*- coding: utf-8 -*-
# * @TODO : None
# * @Note : Convolution Neural Network Structure just like LeNet5
# *        1 pool layer ksize=[1,x,x,1] stride = [1,x,x,1]
# *        2 only full connection weight needs regularization and dropout which
# *          is used while training
# ******************************************************************************
# *
import tensorflow as tf


# define the structure of CNN
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# deep and size of layer
CONV1_DEEP = 32  # the number of filters in layer1
CONV1_SIZE = 5
CONV2_DEEP = 64
CONV2_SIZE = 5
FC_SIZE = 512

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


# define the forward propagation using dropout and arguments
def inference(input_tensor, train, regular,reuse=False):
	"""
	:func  create the hidden layer node using given parameter and Relu
			activate function and calculation the forward propagation
	:param input_tensor: input
	:param train: use dropout when train is true
	:param regular: regularization rate
	:param reuse:  use the weights existed when reuse is true
	:param avg_class: use the Moving average method
	:return: the forward propagation calculation of output
	"""
	# define and calculate the first layer :conv1
	with tf.variable_scope('layer1-conv1',reuse=reuse):
		conv1_weights = get_weight_variable(
			[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],None)
		conv1_biases = tf.get_variable(
			'biases',[CONV1_DEEP],initializer=tf.constant_initializer(0.1))
		conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
		relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

	# define the calculate second layer :pool1
	with tf.name_scope('layer2-pool1'): # use name_scope to distinguish different op
		pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

	# define and calculate the third layer :layer1
	with tf.variable_scope('layer3-conv2',reuse=reuse):
		conv2_weights = get_weight_variable(
			[CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], None)
		conv2_biases = tf.get_variable(
			'biases', [CONV2_DEEP], initializer=tf.constant_initializer(0.1))
		conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1,1,1,1],padding='SAME')
		relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

	# define and calculate fourth layer :pool2
	with tf.name_scope('layer4-pool2'): # use name_scope to distinguish different op
		pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

		# transform the pool2 to the full connection
		pool_shape = pool2.get_shape().as_list()
		# pool_shape[0] is the batch_size of the layer
		fc_nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
		reshaped = tf.reshape(pool2,[pool_shape[0],fc_nodes])

	# define and calculate fifth layer : fc1
	with tf.variable_scope('layer5-fc1',reuse=reuse):
		fc1_weights = get_weight_variable([fc_nodes, FC_SIZE], regular)
		fc1_biases = tf.get_variable(
			'biases', [FC_SIZE], initializer=tf.constant_initializer(0.1))
		fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights) + fc1_biases)
		if train: fc1 = tf.nn.dropout(fc1,0.5)
	# define and calculate sixth layer : fc2
	with tf.variable_scope('layer6-fc2',reuse=reuse):
		fc2_weights = get_weight_variable([FC_SIZE, NUM_LABELS], regular)
		fc2_biases = tf.get_variable(
			'biases', shape=[NUM_LABELS], initializer=tf.constant_initializer(0.1))
		fc2 = tf.matmul(fc1,fc2_weights) + fc2_biases # input softmax layer
	return fc2