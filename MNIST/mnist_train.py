# ***
# ******************************************************************************
# * @filename: The process of train and hyperparameter of NN
# * @author  : tjwx
# * @version :
# * @date    : 2018.09.27
# * @brief   : This file provides all the functions.
# * @reference:Google TensorFlow  practice in chapter 5.5
# ******************************************************************************
# *-*- coding: utf-8 -*-
# * @TODO : None
# * @Note :
# *
# *
# ******************************************************************************
# *

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference

# define the hyperparameter of loss function and Optimizer
learning_rate_base = 0.8
learning_rate_decay = 0.99
regularization_rate = 0.0001
batch_size = 100  # the value usually is 25-256
moving_average_decay = 0.99
training_steps = 30000

# define the path and filename of the model to be saved
model_save_path = "F:/python/Test/MNIST/model"
model_name = "model.ckpt"

# define the loss function and  Optimizer algorithm
def train(mnist):
	"""
		:func : train the NN
		:param mnist:
		:return: None
		"""
	# define input and output
	x = tf.placeholder(tf.float32, shape=[None, mnist_inference.input_node], name='x-input')
	y_ = tf.placeholder(tf.float32, shape=[None, mnist_inference.output_node], name='y-input')

	# define regularization and forward propagation
	global_step = tf.Variable(0, trainable=False)
	regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
	y = mnist_inference.inference(x,None,regularizer,False)

	# apply the moving average to the all trainable parameters
	variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
	variable_averages_op = variable_averages.apply(tf.trainable_variables())

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

	# initialize the save class
	saver = tf.train.Saver()
	# ******** step4: create the session to run the train_op ***********
	with tf.Session() as sess:

		init_op = tf.global_variables_initializer()
		sess.run(init_op)

		# train NN by iter
		for i in range(training_steps):
			xs, ys = mnist.train.next_batch(batch_size)
			_, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

			# save the model every 1000 steps
			if i % 1000 == 0:
				print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
				saver.save(sess,os.path.join(model_save_path,model_name),global_step=global_step)

def main(argv=None):
	mnist = input_data.read_data_sets("F:/python/Test/MNIST/MNIST_temp", one_hot=True)
	train(mnist)


if __name__ == '__main__':
	tf.app.run()


