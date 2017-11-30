# Imports the tensorflow libraries (tf) and input data to train our network
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# =========================================================================== #
#                              DEFINITIONS                                    #
# =========================================================================== #

# Get the data set from input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Creates the nodes for the input images (x) and target output classes (y_)
x = tf.placeholder(tf.float32, shape=[None, 784]) # Input images node
y_ = tf.placeholder(tf.float32, shape=[None, 10]) # Target output class

# Defines the weights and biases functions
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

# Convolves a layer
def conv2d(x, W): return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Pools a layer, reducing its size by half
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# =========================================================================== #
#                         CONVOLUTION MODEL                                   #
# =========================================================================== #

# ========================= #
# First convolutional layer #
# ========================= #

# Creates the weight and bias for the layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# Reshape x to a 4d tensor (2nd and 3rd dimensions are image width and height, 
# 4th is number of color channels)
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Convolves the newly created 4d tensor and reduces its size to 14x14
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# ========================== #
# Second convolutional layer #
# ========================== #

# Creates the weight and bias for the layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# Convolves the newly created pool layer and reduces its size to 7x7
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv1)


# =========== #
# Dense layer #
# =========== #

# Creates the weight and bias for the layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# Reshapes our pooling layer into a batch of vectors then multiplies it by the
# weight, adds the bias, and applies a built in ReLU function
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Reduces overfitting, defines dropout of data
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# ============= #
# Readout layer #
# ============= #

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# =========================================================================== #
#                              TRAINING                                       #
# =========================================================================== #

# Defines our loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
	labels=y_, logits=y_conv))

# Defines our training function
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# =========================================================================== #
#                        EVALUATION OF MODEL                                  #
# =========================================================================== #

# Shows correctly predicted labels
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

# Shows accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	# Initialize the session and variables
	sess.run(tf.global_variables_initializer())

	# Run the training for 20000 iterations
	for i in range(20000):
		batch = mnist.train.next_batch(50)

		if i % 100 == 0:
			train_accuracy = accuracy.eval(feed_dict={
				x: batch[0], y_: batch[1], keep_prob: 1.0})
			print('step %d, training accuracy %g' % (i, train_accuracy))

		train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

	# Prints out resulting data
	# NOTE: Should reach approximately 99.2% accuracy (BETTER)
	print('test accuracy %g' % accuracy.eval(feed_dict={
		x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
