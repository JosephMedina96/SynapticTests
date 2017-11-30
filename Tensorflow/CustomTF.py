# Conceptual information #
# This program is intended to do the following things:
#
# 1. Open an external file containing our data.
# 2. Write the data in the file to a usable medium for machine learning.
# 3. Define labels for if something is a noun or not a noun.
# 4. Train a convolutional neural network to recognize this pattern.
# 
# Upon success with each of these four goals, it will be expanded
# to determine other parts of speech.
#
# This file contains the network itself.

# CustomTF.py #
import tensorflow as tf
import numpy as np
import math

# Model
#
# A neural network in Tensorflow can be very simple or very complicated,
# based on the needs of the developer and the amount of modification that
# is occurring. These networks are built around 'Tensors', or vectors / 
# matrices held in a variety of states.
#
# To start off, we will define a few functions so that initializing new 
# layers will be a bit easier.

# Here we make a function to define a weight variable:
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

# And here, we make a function to generate a bias variable:
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

# We will also want to make some functions to convolve and pool a layer. We will
# be using a convolutional neural network model in order to help increase the accuracy
# of its predictions.

def conv2d(x, W): 
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID', name="conv")

def max_pool_custom(x, sequence_length, filter_size):
	return tf.nn.max_pool(x, ksize=[1, sequence_length - filter_size + 1, 1, 1], 
	strides=[1, 1, 1, 1], padding='VALID', name="pool")

# Having all these functions set up, we are free to begin making our model:
class TextCNN(object):

	def __init__(self, sequence_length, num_classes, vocab_size,
	embedding_size, filter_sizes, num_filters):
		# Placeholders
		# These will hold:
		# - Input data ("input_x")
		# - Our desired outputs ("input_y")
		# - The probability of keeping a node's data ("dropout_keep_prob")
		self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

		# Embedding Layer
		# This layer is named "embedding", and uses the cpu instead of the gpu.
		# It essentially creates a look-up table for vocabulary that is learned from
		# our data.
		with tf.device('/cpu:0'), tf.name_scope("embedding"):
			W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), 
				name="W")

			self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
			self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

		# Convolution Layers
		# These layers are the major portion of our model, and they ultimately will
		# produce one large feature vector.
		pooled_outputs = []
		for i, filter_size in enumerate(filter_sizes):
			with tf.name_scope("conv-maxpool-%s" % filter_size):
				filter_shape = [filter_size, embedding_size, 1, num_filters]
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
				b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
				conv = conv2d(self.embedded_chars_expanded, W)

				h = ReLU(tf.nn.bias_add(conv, b))

				pooled = max_pool_custom(h, sequence_length, filter_size)
				pooled_outputs.append(pooled)

		num_filters_total = num_filters * len(filter_sizes)
		self.h_pool = tf.concat(3, pooled_outputs)
		self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

		# Dropout Layer
		# This layer doesn't really act like a normal layer. It forces
		# each node to learn individually useful features.
		with tf.name_scope("dropout"):
			self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

		# Output Layer
		# This layer makes predictions about new data that enters the system. It does so
		# by using matrix multiplication derived from the scores it gets after being trained.
		with tf.name_scope("output"):
			W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1),
				name="W")
			b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
			self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
			self.predictions = tf.argmax(self.scores, 1, name="predictions")

		# Loss Layer
		# This layer doesn't function like a normal layer. Instead, it holds the data
		# that determines the rate of loss of data from our model's nodes.
		with tf.name_scope("loss"):
			losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.scores, 
				logits=self.input_y)
			self.loss = tf.reduce_mean(losses)

		# Accuracy Layer
		# This layer is useful for training and testing purposes, as it returns the 
		# accuracy of the system.
		with tf.name_scope("accuracy"):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), 
				name="accuracy")
