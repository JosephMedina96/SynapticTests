# Imports the tensorflow libraries (tf) and input data to train our network
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# =========================================================================== #
#                              DEFINITIONS                                    #
# =========================================================================== #

# Get the data set from input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Create an interactive session from the tensorflow libraries
sess = tf.InteractiveSession()

# Creates the nodes for the input images (x) and target output classes (y_)
x = tf.placeholder(tf.float32, shape=[None, 784]) # Input images node
y_ = tf.placeholder(tf.float32, shape=[None, 10]) # Target output class

# Defines the weights (W) and biases (b)
W = tf.Variable(tf.zeros([784, 10])) # Weights
b = tf.Variable(tf.zeros([10]))      # Biases

# =========================================================================== #
# Before a variable can be used in a tensorflow network, it must be initialized.
# =========================================================================== #

# Initializes our variables
sess.run(tf.global_variables_initializer())

# =========================================================================== #
#                           REGRESSION MODEL                                  #
# =========================================================================== #

# Actual model
y = tf.matmul(x, W) + b # Multiplies vectorized input images by the weight matrix
                        # then adds the bias

# Specifies the loss function
cross_entropy = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# =========================================================================== #
#                              TRAINING                                       #
# =========================================================================== #

# Utilizes a built-in training method of the tensorflow libraries
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(1000):
	batch = mnist.train.next_batch(100)
	train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# =========================================================================== #
#                        EVALUATION OF MODEL                                  #
# =========================================================================== #

# Shows correctly predicted labels
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# Shows accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Prints out resulting data
# NOTE: Should reach approximately 92% accuracy (BAD)
print("============")
print("Accuracy:")
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
