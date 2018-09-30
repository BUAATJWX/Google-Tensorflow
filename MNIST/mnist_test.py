# ***
# ******************************************************************************
# * @filename: The process of test the model saved by the mnist_train.py
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

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

# load the lastest model every 10s and evaluate the accuracy on the test data
eval_interval_secs = 10

# test the model
def evaluate(mnist):
	with tf.Graph().as_default() as g:
		# define input and output
		x = tf.placeholder(tf.float32, shape=[None, mnist_inference.input_node], name='x-input')
		y_ = tf.placeholder(tf.float32, shape=[None, mnist_inference.output_node], name='y-input')

		# forward propagation process
		y = mnist_inference.inference(x, None,None,False)

		# define accuracy
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

		# get the moving average weights saved in mnist_train.py by rename way
		variable_averages = tf.train.ExponentialMovingAverage(mnist_train.moving_average_decay)
		variable_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variable_to_restore)

		while True:
			with tf.Session() as sess:
				ckpt = tf.train.get_checkpoint_state(mnist_train.model_save_path)
				if ckpt and ckpt.model_checkpoint_path:
					saver.restore(sess,ckpt.model_checkpoint_path)
					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
					accuracy_score = sess.run(accuracy,feed_dict=validate_feed)
					print("After %s training step(s),validation accuracy = %g "%
						  (global_step, accuracy_score))
				else:
					print("No checkpoint file found")
					return
			time.sleep(eval_interval_secs) # sleep the thread every interval
def main(argv=None):
	mnist = input_data.read_data_sets("F:/python/Test/MNIST/MNIST_temp", one_hot=True)
	evaluate(mnist)


if __name__ == '__main__':
	tf.app.run()
