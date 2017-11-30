# Imports #
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import collections, socket, math, os, random, zipfile
from six.moves import xrange

# Converts necessary data into a message to interact with TouchDesigner
def convertToMessage(input_text, output):
	WORDINPUT = input_text # User input text
	WORDOUTPUT = output    # System output text

	input_number = convertToNumber(WORDINPUT)
	output_number_main = convertToNumber(WORDOUTPUT)

	print("Input number: " + input_number)
	print("Output number main: " + output_number_main)
	print("")

	# Maps the input number to an integer between 2 and 6 (CONTROLS SHAPE)
	input_map = mapNumberToRange(input_number, -10000000, 10000000, 2, 6)
	input_map_fixed = round(input_map)

	# Maps the output number to a float between 0 and 1 (CONTROLS COLOR)
	output_map = mapNumberToRange(output_number_main, -10000000, 10000000, 0, 1)
	output_map_fixed = round(output_map, 2)

	print("Mapped Input: " + input_map_fixed)
	print("Mapped Output: " + output_map_fixed)
	print("")

	r = math.ceil(output_map_fixed)
	g = output_map_fixed
	b = math.floor(output_map_fixed)

	print("R: " + r + " G: " + g + " B: " + b)
	print("")

	# Creates an associated number (CONTROLS PARTICLE FLOW)
	ASSOCIATED = math.floor(input_map_fixed + output_map_fixed)
	if (ASSOCIATED % 3 == 0):
		ASSOCIATED = 3
	else:
		ASSOCIATED = 0

	string = input_map_fixed + ", " + r + ", " + g + ", " + b + ", " + ASSOCIATED
	return string

# Converts a string to a number
def convertToNumber(string):
	return int(string).from_bytes(string.encode(), 'little')

# Creates a batch of data to be used for training or testing
def createBatch(batch_size, num_skips, skip_window):
	global data_index
	assert batch_size % num_skips == 0
	assert num_skips <= 2 * skip_window

	batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	span = 2 * skip_window + 1
	buffer = collections.deque(maxlen=span)

	if data_index + span > len(data):
		data_index = 0

	buffer.extend(data[data_index: data_index + span])
	data_index += span

	for i in range(batch_size // num_skips):
		context_words = [w for w in range(span) if w != skip_window]
		words_to_use = random.sample(context_words, num_skips)

		for j, context_word in enumerate(words_to_use):
			batch[i * num_skips + j] = buffer[skip_window]
			labels[i * num_skips + j, 0] = buffer[context_word]

		if data_index == len(data):
			for word in data[:span]:
				buffer.append(word)
		else:
			buffer.append(data[data_index])
			data_index += 1

	data_index = (data_index + len(data) - span) % len(data)
	return batch, labels

# Creates a bias for the given vocabulary size
def createBias(vocabulary_size):
	bias = tf.Variable(tf.zeros([vocabulary_size]))
	return bias

# Creates a data set to test the model
def createDataSet(words, n_words):
	count = [['UNK', -1]]
	count.extend(collections.Counter(words).most_common(n_words - 1))
	dictionary = dict()

	for word, _ in count:
		dictionary[word] = len(dictionary)

	data = list()
	unk_count = 0

	for word in words:
		index = dictionary.get(word, 0)

		if index == 0:
			unk_count += 1

		data.append(index)

	count[0][1] = unk_count
	reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data, count, dictionary, reversed_dictionary

# Creates the embedding layer
def createEmbeddings(vocabulary_size, embedding_size):
	embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
	return embeddings

# Creates a weight based on the given vocabulary size and embedding size
def createWeight(vocabulary_size, embedding_size):
	weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], 
		stddev=1.0 / math.sqrt(embedding_size)))
	return weights

def defineElements(valid_examples):
	graph = tf.Graph()

	# Graph details
	with graph.as_default():

		train_inputs = tf.placeholder(tf.int32, shape=[batch_size]) # inputs
		train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1]) # labels
		vaild_dataset = tf.constant(valid_examples, dtype=tf.int32) # dataset

		# Done on the CPU as to avoid creating errors
		with tf.device('/cpu:0'):
			# Generates the embedding layer
			embeddings = createEmbeddings(vocabulary_size, embedding_size)
			embed = tf.nn.embedding_lookup(embeddings, train_inputs)

			weights = createWeight(vocabulary_size, embedding_size)
			bias = createBias(vocabulary_size)

		# Defines the graph's loss
		loss = tf.reduce_mean(tf.nn.nce_loss(weights=weights,
			biases=bias, labels=train_labels, inputs=embed, num_sampled=num_sampled,
			num_classes=vocabulary_size))

		# Utilizes a built-in optimizer to help with accuracy
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

		# Normalizes and validates the embeddings from the embedding layer
		norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
		normalized_embeddings = embeddings / norm
		valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, vaild_dataset)

		similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

		return loss, optimizer, similarity

# Sends data through the network
def evaluate(test_input):
	test_input.append(NULL_VAL)

	for step in xrange(num_steps):
		train_inputs = tf.placeholder(tf.int32, shape=[batch_size]) # inputs
		train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1]) # labels
		batch_inputs, batch_labels = createBatch(batch_size, num_skips, skip_window)
		feed_dict = {train_inputs: test_input, train_labels: batch_labels}

		_, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
		
		# Every 10th iteration, print the result
		if step % 10 == 0:
			sim = similarity.eval()

			for i in xrange(valid_size):
				valid_word = reverse_dictionary[valid_examples[i]]
				top_k = 1
				nearest = (-sim[i, :]).argsort()[1:top_k + 1]
				log_str = 'Nearest to (%s):' % valid_word

				for k in xrange(top_k):
					close_word = reverse_dictionary[nearest[k]]
					log_str = '%s (%s),' % (log_str, close_word)

				print(log_str)
				print("===========================")
				print("")

				return close_word

# Verifies that a file exists
def find_file(filename):
	if os.path.isfile(filename):
		print("File found!")
		print("===========================")
		local_filename = filename
		return local_filename
	else:
		print("File not found.")
		print("===========================")
		raise ValueError("Could not find file: ", filename)

# Maps a number from one range to a new one
def mapNumberToRange(num, in_min, in_max, out_min, out_max):
	return (num - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# Reads zipfiles as text
def read_data(filename):
	with zipfile.ZipFile(filename) as f:
		data = tf.compat.as_str(f.read(f.namelist()[0])).split()
	return data

# Sending data to TouchDesigner
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
MESSAGE = "N/A"

print("UDP target IP: " + str(UDP_IP))
print("UDP target port: " + str(UDP_PORT))
print("message: " + str(MESSAGE))

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP

# Variables #
NULL_VAL = 0

directory = "data.zip" # Holds the name of the .zip file containing data
filename = find_file(directory) # Verifies that the .zip file exists
vocabulary = read_data(filename) # Creates a vocabulary to teach the system

vocabulary_size = 260 # Size of the dataset
embedding_size = 128

# Creates the dataset
data, count, dictionary, reverse_dictionary = createDataSet(vocabulary, vocabulary_size)
del vocabulary # Deletes the vocabulary to save memory

print('Most Common words (+UNK)', count[:5])
print("===========================")
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
print("===========================")

data_index = 0 # Used to specify what piece of data the system is currently on
batch_size = 2
num_sampled = 64
num_skips = 2
skip_window = 1

# Creates a test batch
batch, labels = createBatch(batch_size=2, num_skips=2, skip_window=1)
for i in range(2):
	print(batch[i], reverse_dictionary[batch[i]], '->', 
		labels[i, 0], reverse_dictionary[labels[i, 0]])

valid_size = 16
valid_window = 260
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

# For Loading the model:
export_name = os.path.dirname(__file__)
export_directory = os.path.join(export_name, 
'/users/josephmedina/SynapticTests/Tensorflow/Builds/Skip-Gram')

num_steps = 10 # Number of test iterations

with tf.Session(graph=tf.Graph()) as sess:

	# Load the model
	graph = tf.saved_model.loader.load(sess, ["TRAINING"], export_directory)
	print("===========================")
	print('Model Successfully Loaded!')
	print("===========================")
	print("")

	loss, optimizer, similarity = defineElements(valid_examples)

	# Keep taking inputs until we want to stop
	while True:
		print("======================")
		test_input = raw_input("Type something here: ")
		print("======================")
		print("")

		test_number = int(test_input)

		# Evaluate the input
		result = evaluate(test_number)
		MESSAGE = convertToMessage(test_input, result)
		sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
		while True:
			keep_going = raw_input('Keep going? (Yes/No): ')
			answer = keep_going[:1].lower()
		
			if answer in ('y', 'n'):
				break
		
			print("Invalid Input.")
	
		if answer == "y":
			print("=================================")
			continue
		else:
			break
