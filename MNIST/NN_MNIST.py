# ***
# ******************************************************************************
# * @filename: A complete NN for MNIST recognition
# * @author  : tjwx
# * @version :
# * @date    : 2018.09.27
# * @brief   : This file provides all the functions.
# * @reference:Google TensorFlow  practice in chapter 5.1
# ******************************************************************************
# *-*- coding: utf-8 -*-
# * @TODO : None
# * @Note : Network Structure
# *         input ；Tensor[n,748] , hidden layer : [500] output:Tensor [n,10]
# *
# ******************************************************************************
# *

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# ******** step1: define hyperparameter of NN *************

# the structure of NN
input_node = 784
output_node = 10
layer1_node = 500
# the hyperparameter of loss function and Optimizer
learning_rate_base = 0.8
learning_rate_decay = 0.99
regularization_rate = 0.0001
batch_size = 100  # the value usually is 25-256
moving_average_decay = 0.99
training_steps = 30000
tf.set_random_seed(1)

# ******** step2: define the NN model structure and parameters *****
# define real input and output variable
def get_weight(shape, Lamada):
	"""
	:func  generate the weights and corresponding regularization  which will be added
			to the collection 'losses'
	:param shape: the shape of weights to be created
	:param Lamada: the regularization rate
	:return: the weights
	"""
	weights = tf.get_variable('weights', shape=shape,
							  initializer=tf.truncated_normal_initializer(stddev=0.1))
	if Lamada != None:
		tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(Lamada)(weights))
	return weights


x = tf.placeholder(tf.float32, shape=[None, input_node], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[None, output_node], name='y-input')

global_step = tf.Variable(0, trainable=False)


def inference(input_tensor, avg_class, reuse=False):
	"""
	:func  create the hidden layer node using given parameter and Relu
			activate function and calculation the forward propagation
	:param input_tensor:
	:param avg_class:
	:param weights1:
	:param weights2:
	:param biases1:
	:param biases2:
	:return: the forward propagation calculation of output
	"""
	# define the first layer
	with tf.variable_scope('layer1', reuse=reuse):
		weights = get_weight([input_node, layer1_node], regularization_rate)
		biases = tf.get_variable('biases',
								 shape=[layer1_node], initializer=tf.constant_initializer(1.0))
		if avg_class == None:
			layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
		else:
			layer1 = tf.nn.relu(
				tf.matmul(input_tensor, avg_class.average(weights)) + avg_class.average(biases))

	# define the second layer
	with tf.variable_scope('layer2', reuse=reuse):
		weights = get_weight([layer1_node, output_node], regularization_rate)
		biases = tf.get_variable('biases',
								 shape=[output_node], initializer=tf.constant_initializer(1.0))
		if avg_class == None:
			layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)
		else:
			layer2 = tf.nn.relu(
				tf.matmul(layer1, avg_class.average(weights)) + avg_class.average(biases))
	return layer2

# ******** step3: define the loss function and  Optimizer algorithm *****

def train(mnist):
	"""
	:func : train the NN
	:param mnist:
	:return: None
	"""

	y = inference(x, None)

	# apply the moving average to the all trainable parameters
	variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
	variable_averages_op = variable_averages.apply(tf.trainable_variables())
	average_y = inference(x, variable_averages,True)
	# loss function
	cross = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
	cross_entropy_loss = tf.reduce_mean(cross)
	# regularization
	tf.add_to_collection('losses', cross_entropy_loss)
	loss = tf.add_n(tf.get_collection('losses'))

	# define optimize arguments and train process
	learning_rate = tf.train.exponential_decay(
		learning_rate_base,
		global_step,
		mnist.train.num_examples / batch_size,
		learning_rate_decay
	)
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
	with tf.control_dependencies([train_step, variable_averages_op]):
		train_op = tf.no_op(name='train')
	# define the acc in validation data and test
	correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
	acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# ******** step4: create the session to run the train_op ***********
	with tf.Session() as sess:

		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		# 初始化所有参数，同上面两句作用一致
		# tf.initialize_all_variables().run()
		# tf.global_variables_initializer().run()
		# 准备验证数据，一般在神经网络的训练过程中会通过验证数据来判断大致停止的条件和评判训练的效果
		validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
		# 准备测试数据，在实际中，这部分数据在训练时是不可见的，这个数据只是作为模型优劣的最后评价标准
		test_feed = {x: mnist.test.images, y_: mnist.test.labels}
		# 迭代的训练神经网络
		for i in range(training_steps):
			xs, ys = mnist.train.next_batch(batch_size)
			_, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
			if i % 1000 == 0:
				print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
				validate_acc = sess.run(acc, feed_dict=validate_feed)
				print("After %d training step(s),validation accuracy using average model is %g " % (step, validate_acc))
				test_acc = sess.run(acc, feed_dict=test_feed)
				print("After %d training step(s) testing accuracy using average model is %g" % (step, test_acc))


def main(argv=None):
	mnist = input_data.read_data_sets("F:/python/Test/MNIST/MNIST_temp", one_hot=True)
	train(mnist)


if __name__ == '__main__':
	tf.app.run()
