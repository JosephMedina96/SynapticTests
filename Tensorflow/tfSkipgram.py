# tfSkipgram.py

# IMPORTANT INFORMATION #
# This file generates a trained skip-gram model, which takes the words that it is given
# and learns patterns between them.
#
# If you have already generated a model, either:
# 1. Delete the old model and use the current directory
#                   --OR--
# 2. Find the export directory variable and specify a new one

# Imports #
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import collections, math, os, random, zipfile
from tempfile import gettempdir
from six.moves import xrange

# Useful functions #
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

# Creates a data set to train the model
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

# Reads zipfiles as text
def read_data(filename):
	with zipfile.ZipFile(filename) as f:
		data = tf.compat.as_str(f.read(f.namelist()[0])).split()
	return data

# Definitions #
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
batch_size = 130
num_sampled = 64
num_skips = 2
skip_window = 1

# Creates a test batch
batch, labels = createBatch(batch_size=130, num_skips=2, skip_window=1)
for i in range(130):
	print(batch[i], reverse_dictionary[batch[i]], '->', 
		labels[i, 0], reverse_dictionary[labels[i, 0]])

print("===========================")

valid_size = 16
valid_window = 260
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

# Model #
graph = tf.Graph()

# Graph details
with graph.as_graph_def():

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

	init = tf.global_variables_initializer()

num_steps = 100001 # Number of training iterations

# For saving the model:
export_name = os.path.dirname(__file__)
export_directory = os.path.join(export_name, 
	'/users/josephmedina/SynapticTests/Tensorflow/Builds/Skip-Gram')
saved_model = tf.saved_model.builder.SavedModelBuilder(export_directory)

# Session details
with tf.Session(graph=graph) as session:

	init.run()
	print("===========================")
	print('Initialized...')
	print("===========================")

	average_loss = 0

	for step in xrange(num_steps):
		batch_inputs, batch_labels = createBatch(batch_size, num_skips, skip_window)
		feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

		_, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
		average_loss += loss_val

		# Every 2000th iteration, print the average loss
		if step % 2000 == 0:
			if step > 0:
				average_loss /= 2000

			print('Average loss at step ', step, ': ', average_loss)
			print("===========================")
			average_loss = 0

		# Every 10000th iteration, print the patterns being learned from 16 random instances
		if step % 10000 == 0:
			sim = similarity.eval()

			for i in xrange(valid_size):
				valid_word = reverse_dictionary[valid_examples[i]]
				top_k = 8
				nearest = (-sim[i, :]).argsort()[1:top_k + 1]
				log_str = 'Nearest to (%s):' % valid_word

				for k in xrange(top_k):
					close_word = reverse_dictionary[nearest[k]]
					log_str = '%s (%s),' % (log_str, close_word)

				print(log_str)
				print("===========================")
				print("")

	print("***===***===***===***")
	print("Training Complete!")
	print("***===***===***===***")
	print("")
	final_embeddings = normalized_embeddings.eval()

	# Add the trained portion of the model to the export
	saved_model.add_meta_graph_and_variables(session, ["TRAINING"])

# Adds the graph of the model to the export
with tf.Session(graph=graph) as sess:
	saved_model.add_meta_graph(["SERVING"])

saved_model.save()
