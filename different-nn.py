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
	def __init__(self, x, y1, y2, bias, leraningRate=0.1):
		self.input = x
		self.weights1 = np.array([[0.0], [0.0]])
		self.weights2 = np.array([[0.0], [0.0]])

		sys.stdout.write('\rweights: {} {} {} {}\n'.format(self.weights1[0], self.weights1[1], self.weights2[0], self.weights2[1]))
		self.y1 = y1
		self.y2 = y2
		self.output1 = np.zeros(y1.shape)
		self.output2 = np.zeros(y2.shape)
		self.leraningRate = leraningRate
		self.bias = bias

	def feedForward(self):
		self.output1 = sigmoid(net(self.input, self.weights1, self.bias))
		self.output2 = sigmoid(net(self.input, self.weights2, self.bias))

	def backPropagation(self):
		d_weights1 = (-self.leraningRate) * np.dot(self.input.T, np.multiply(E_(self.output1, self.y1), sigmoid_(self.output1)))
		d_weights2 = (-self.leraningRate) * np.dot(self.input.T, np.multiply(E_(self.output2, self.y2), sigmoid_(self.output2)))

		self.weights1 += d_weights1
		self.weights2 += d_weights2

	def predict(self, x):
		return (sigmoid(net(x, self.weights1, self.bias)), sigmoid(net(x, self.weights2, self.bias)))

if __name__ == '__main__':
	x = np.array([
		[0, 0],
		[0, 1],
		[1, 0],
		[1, 1]
	])
	y1 = np.array([
		[0],
		[1],
		[1],
		[1]
	])
	y2 = np.array([
		[1],
		[0],
		[0],
		[0]
	])
	bias = -1

	maxIter = 500000
	iterStep = maxIter / 10
	learningRate = 0.1

	nn = NeuralNetwork(x, y1, y2, bias, learningRate)

	for iter in range(maxIter):
		nn.feedForward()
		nn.backPropagation()
		if ((iter + 1) % iterStep == 0):
			sys.stdout.write('\rweights: {} {} {} {}\n'.format(nn.weights1[0], nn.weights1[1], nn.weights2[0], nn.weights2[1]))
		sys.stdout.write('\riter: {}%'.format(((iter + 1) * 100) / maxIter))

	print('\n\nafter training:')
	for i in range(4):
		print(x[i], y1[i], nn.output1[i], y2[i], nn.output2[i])

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
