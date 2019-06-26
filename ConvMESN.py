from __future__ import print_function

import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config = config))

import numpy as np
import cPickle as cp
import random

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, GlobalMaxPooling2D, concatenate

import Reservoir #the code for the reservoir with skip connection
import utils

print('\nLoading data...')
filepath = './datasets/ASD.p'
samples, labels, _ = cp.load(open(filepath, 'rb'))
labels, num_classes = utils.transformer(labels)

#Hyper-parameters for Multi-timescale Memory Encoder
num_samples, time_length, n_in = samples.shape #the number of input units
n_res = n_in * 3 #the number of reservoir units
IS = 0.1 #the input scaling
SR = 0.1 #the spectral radius
SP = 0.5 #the sparsity
skip_lengths = [1, 3, 9, 27] #the skip lengths of reserviors
num_reservoirs = len(skip_lengths) #the number of reservoirs

print('Getting echo states...\n')
esns = [Reservoir.reservoir(n_in, n_res, IS, SR, SP) for i in range(num_reservoirs)]
echo_states = np.empty((num_samples, num_reservoirs, time_length, n_res), np.float32)
for i in range(num_reservoirs):
	echo_states[:,i,:,:] = esns[i].get_echo_states(samples, skip_lengths[i])

#N-fold cross validation
num_folds = 10
li = range(num_samples)
random.shuffle(li)
folds = []
num_samples_per_fold = num_samples / num_folds
p = 0
for f in range(num_folds-1):
	folds.append(li[p:p+num_samples_per_fold])
	p += num_samples_per_fold
folds.append(li[p:])

#Hyper-parameters for Convolutional Memory Learner
input_shape = (1, time_length, n_res)
num_types = 3 #the number of height types of filters 
nb_filter = num_classes * 2 #the number of filters for each height (K)
nb_row = [1, 3, 9, 27] #the heights of filters 
nb_col = n_res #the widths of filters
kernel_initializer = 'lecun_uniform'
activation = 'relu'
padding = 'valid'
strides = (1, 1)
data_format = 'channels_first'

#Hyper-parameters for training model
optimizer = 'adam'
loss = 'categorical_crossentropy'
batch_size = 88
nb_epoch = 30
verbose = 1

accs = [0 for f in range(num_folds)]

for f in range(num_folds):
    
	inputs = []
	multi_pools = []

	for i in range(num_reservoirs):
	
		input = Input(shape = input_shape)
		inputs.append(input)
	
		pools = []

		for j in range(num_types):
            #the convolutional layer
			conv = Conv2D(nb_filter, (nb_row[i] * (j+1), nb_col), kernel_initializer = kernel_initializer, activation = activation, padding = padding, strides = strides, data_format = data_format)(input)
			pool = GlobalMaxPooling2D(data_format = data_format)(conv)
			pools.append(pool)
	
		multi_pools.append(Dense(nb_filter, kernel_initializer = kernel_initializer, activation = activation)(concatenate(pools)))

	#the fully-connected layer
	features = Dense(nb_filter * num_reservoirs / 2, kernel_initializer = kernel_initializer, activation = activation)(concatenate(multi_pools))

	#the softmax layer
	outputs = Dense(num_classes, kernel_initializer = kernel_initializer, activation = 'softmax')(features)

	model = Model(inputs = inputs, outputs = outputs)
	model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])

    #N-fold cross validation
	num_samples_test = len(folds[f])
	num_samples_train = num_samples - num_samples_test

	echo_states_test = np.empty((num_samples_test, num_reservoirs, time_length, n_res), np.float32)
	labels_test = np.empty((num_samples_test, num_classes))
	for i in range(num_samples_test):
		echo_states_test[i] = echo_states[folds[f][i]]
		labels_test[i] = labels[folds[f][i]]
	
	echo_states_train = np.empty((num_samples_train, num_reservoirs, time_length, n_res), np.float32)
	labels_train = np.empty((num_samples_train, num_classes))
	p = 0
	for i in range(num_folds):
		if i != f:
			for j in range(len(folds[i])):
				echo_states_train[p] = echo_states[folds[i][j]]
				labels_train[p] = labels[folds[i][j]]
				p += 1
	
	echo_states_train = [echo_states_train[:,i:i+1,:,:] for i in range(num_reservoirs)]
	echo_states_test = [echo_states_test[:,i:i+1,:,:] for i in range(num_reservoirs)]
	
    #training and testing the model
	history = model.fit(echo_states_train, labels_train, batch_size = batch_size, epochs = nb_epoch, verbose = verbose, validation_data = (echo_states_test, labels_test))
	
	for acc in history.history['val_acc']:
		if acc > accs[f]:
			accs[f] = acc
	
	print('Fold %d/%d' % (f+1, num_folds))
	print('Accuracy: %f\n' % (accs[f]))

#print the final mean accuracy for N-fold cross validation
print('Mean Accuracy: %f' % (sum(accs)/num_folds))
