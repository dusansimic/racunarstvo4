import numpy as np
import time
import sys

# Sigmoidna funkcija
# Izgleda je python dovoljno pametan da kad mu se baci niz da on racuna odvojeno stvari
def sigmoid(x, deriv=False):
	if (deriv):
		return x * (1 - x)
	return 1 / (1 + np.exp(-x))

def think(input):
	output = sigmoid(np.dot(input, W0))
	return output

# Input
# Napomena: prve dve vrednosti su inputi za OR gate a treca je bias
X = np.array([
	[0,0,1],
	[0,1,1],
	[1,0,1],
	[1,1,1]
])

# Tacni rezultati
Y = np.array([
	[0],
	[1],
	[1],
	[1]
])

numberInputs = 3
maxIter = 500000
iterStep = maxIter / 10
learningRate = 0.1

np.random.seed(int(time.time()))

# Generisanje random tezina
W0 = 2 * np.random.random((numberInputs, 1)) - 1
sys.stdout.write('\rweights: {} {} {}\n'.format(W0[0], W0[1], W0[2]))

for iter in range(maxIter):
	# Layer 0 (input)
	L0 = X
	# Layer 1 (hidden)
	L1 = sigmoid(np.dot(L0, W0))

	# Greske u layeru 1
	L1_error = Y - L1

	# Delta koji treba dodati na tezine
	L1_delta = L1_error * sigmoid(L1, deriv=True)

	W0 += learningRate * np.dot(L0.T, L1_delta)

	# Lepi ispisi da bi se videlo kako se tezine menjaju i kolko je ostalo do kraja :3
	if ((iter+1) % iterStep == 0):
		sys.stdout.write('\rweights: {} {} {}\n'.format(W0[0], W0[1], W0[2]))
	sys.stdout.write('\riter: {}%'.format(((iter + 1) * 100) / maxIter))

# Nakon treninga ispisi sta je istrenirao
print('\n\nafter training:')
for i in range(4):
	#
	print(X[i], Y[i], L1[i])

input()

# Testiraj mrezu
testInput = np.array([
	[0,0,1],
	[1,0,1],
	[0,1,1],
	[1,1,1],
	[0,0,0],
	[1,0,0],
	[0,1,0],
	[1,1,0]
])
print('test:')
for test in testInput:
	print(test, think(test))