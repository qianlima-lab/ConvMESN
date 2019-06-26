import numpy as np
from keras.utils import np_utils

#a tool for transforming labels into the format that Keras can read in
def transformer(labels):

	indexes = np.unique(labels)
	num_classes = indexes.shape[0]
	num_samples = labels.shape[0]

	for i in range(num_samples):
		new_label = np.argwhere(indexes == labels[i])[0][0]
		labels[i] = new_label
	labels = np_utils.to_categorical(labels, num_classes)

	return labels, num_classes
