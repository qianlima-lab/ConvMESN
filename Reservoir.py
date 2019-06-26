import numpy as np
import scipy as sp
from scipy.sparse import *
from scipy.sparse.linalg import *

class reservoir(object):
	
    #initializing a reservoir
	def __init__(self, n_in, n_res, IS, SR, SP, leakyrate = 1.0, use_bias = False):
		
		self.n_in = n_in
		self.n_res = n_res
		self.IS = IS
		self.SR = SR
		self.SP = SP
		self.leakyrate = leakyrate
		self.use_bias = use_bias

		self.W_in = 2 * np.random.random(size = (self.n_res, self.n_in)) - 1

		W_res_temp = sp.sparse.rand(self.n_res, self.n_res, self.SP)
		vals, vecs = sp.sparse.linalg.eigsh(W_res_temp, k = 1)
		self.W_res = (self.SR * W_res_temp / vals[0]).toarray()

		b_bound = 0.1
		self.b = 2 * b_bound * np.random.random(size = (self.n_res)) - b_bound
	
    #getting echo states by a reservoir with skip connection
	def get_echo_states(self, samples, skip_length):
		
		num_samples, time_length, _ = samples.shape
		echo_states = np.empty((num_samples, time_length, self.n_res), np.float32)

		for i in range(num_samples):
			
			collect_states = np.empty((time_length, self.n_res), np.float32)
			x = [np.zeros((self.n_res)) for n in range(skip_length)]

			for t in range(time_length):
				
				u = samples[i, t]
				index = t % skip_length

				if self.use_bias:
					xUpd = np.tanh(np.dot(self.W_in, self.IS * u) + np.dot(self.W_res, x[index]) + self.b)
				else:
					xUpd = np.tanh(np.dot(self.W_in, self.IS * u) + np.dot(self.W_res, x[index]))

				x[index] = (1 - self.leakyrate) * x[index] + self.leakyrate * xUpd

				collect_states[t] = x[index]

			echo_states[i] = collect_states

		return echo_states
