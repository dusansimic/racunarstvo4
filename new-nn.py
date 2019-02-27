import numpy as np
import sys
import time

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_(x):
	return x * (1 - x)

def E_(y, t):
	return y - t

def net(inputs, weights, bias):
	return np.subtract(np.dot(inputs, weights), bias)

class NeuralNetwork:
	def __init__(self, x, y, bias, learningRate=1):
		self.input = x
		self.weights = np.array([[0.0], [0.0]])
		#self.weights = np.random.random(0,1)
		sys.stdout.write('\rweights: {} {}\n'.format(self.weights[0], self.weights[1]))
		self.y = y
		self.output = np.zeros(y.shape)
		self.learningRate = learningRate
		self.bias = bias

	def feedForward(self):
		self.output = sigmoid(net(self.input, self.weights, self.bias))
	
	def backPropagation(self):
		d_weights = (-self.learningRate) * np.dot(self.input.T, np.multiply(E_(self.output, self.y), sigmoid_(self.output)))

		self.weights += d_weights

	def predict(self, x):
		return sigmoid(net(x, self.weights, self.bias))

if __name__ == '__main__':
	x = np.array([
		[0, 0],
		[0, 1],
		[1, 0],
		[1, 1]
	])
	y = np.array([
		[0],
		[1],
		[1],
		[1]
	])
	bias = 1

	maxIter = 500000
	iterStep = maxIter / 10
	learningRate = 0.1

	nn = NeuralNetwork(x, y, bias, learningRate)

	for iter in range(maxIter):
		nn.feedForward()
		nn.backPropagation()
		if ((iter + 1) % iterStep == 0):
			sys.stdout.write('\rweights: {} {}\n'.format(nn.weights[0], nn.weights[1]))
		sys.stdout.write('\riter: {}%'.format(((iter + 1) * 100) / maxIter))

	print('\n\nafter training:')
	for i in range(4):
		print(x[i], y[i], nn.output[i])

	print('\n\n')
	res = nn.predict(np.array([
		[0, 1],
		[1, 0],
		[0, 0],
		[1, 1],
		[0, 0],
		[0, 1],
		[1, 1]
	]))
	for result in res:
		print(result, 1 if result[0] > 0.5 else 0 if result[0] < 0.5 else '?')
