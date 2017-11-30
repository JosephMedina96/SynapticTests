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
# This file trains the network by loading our data and labels into it.

# TrainCNN.py #
import tensorflow as tf
import numpy as np
import pandas as pd
import os, sys, json, time, logging
from CustomTF import conv2d, max_pool_custom, TextCNN
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split

logging.getLogger().setLevel(logging.INFO)

# Our first goal is to create a function that can load our data and labels.
def load_data_and_labels(filename):
	# Load data and labels
	df = pd.read_csv(filename, compression='zip', dtype={'Word': object})
	selected = ['Part', 'Word']
	non_selected = list(set(df.columns) - set(selected))

	df = df.drop(non_selected, axis=1) # Drops any non selected column
	df = df.dropna(axis=0, how='any', subset=selected) # Drops null rows
	df = df.reindex(np.random.permutation(df.index)) # Shuffles the dataframe

	# Map the data labels to one hot labels
	labels = sorted(list(set(df[selected[0]].tolist())))
	one_hot = np.zeros((len(labels), len(labels)), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))

	x_raw = df[selected[1]].tolist()
	y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
	return x_raw, y_raw, df, labels

# The training will be handled by the following function:
def train_cnn():
	# First, before anything can be done the data, the labels, and the training parameters
	# must be loaded into our model. 

