'''
This is a class implementation of a neural network. The activation function
used is sigmoid for all layers. It's possible to use gradient descent or 
stochastic gradient descent to train the network.

To create a network just pass the number of neurons in each layer in the 
constructor. For example:

my_network = NeuralNetwork(120, 30, 10)

You'll create a 3-layer network (input layer, hidden layer, output layer) 
containing 120 neuros in the input, 30 neurons in the hidden layer and 10 
neurons in the output. You can see a working example in the "main.py" file.

For a straightforward implementation, please check the "simple_example.py" 
file.
'''

import numpy as np
import random


class NeuralNetwork(object):
    def __init__(self, *sizes):
        self.num_layers = len(sizes)
        self.b = [np.random.uniform(-0.12, 0.12, (1, n)) for n in sizes[1:]]
        self.w = [np.random.uniform(-0.12, 0.12, (n, m)) for n, m in zip(sizes[:-1], sizes[1:])]
        # self.b = [ np.random.randn(1, n) for n in sizes[1:] ]
        # self.w = [ np.random.randn(n, m) for n, m in zip(sizes[:-1], sizes[1:]) ]
        self.a = [[] for i in xrange(self.num_layers)]
        self.z = [[] for i in xrange(self.num_layers-1)]

        self.training_size = 0
        self.reg_lambda = 1.0

        self.grad_w = [[] for i in xrange(self.num_layers-1)]
        self.grad_b = [[] for i in xrange(self.num_layers-1)]

    def feedForward(self, X):
        self.a[0] = X
        for i in xrange(self.num_layers-1):
            self.z[i] = self.a[i].dot(self.w[i]) + self.b[i]
            self.a[i+1] = self.sigmoid(self.z[i])

    def cost(self, y):
        diff_Ks = -y * np.log(self.a[-1]) - (1 - y) * np.log(1 - self.a[-1])
        J = sum(np.sum(diff_Ks, axis=0)) / self.training_size

        # regularization
        totalSum = np.array([])
        for w in self.w:
            totalSum = np.concatenate((totalSum, sum(w ** 2.0)))
        totalSum = sum(totalSum)
        J = J + ((self.reg_lambda * totalSum) / (2.0 * self.training_size))

        return J

    def backpropagation(self, y):
        batch_size = float(len(y))
        delta = (self.a[-1] - y)
        self.grad_w[-1] = ((delta.T).dot(self.a[-2])) / batch_size
        self.grad_b[-1] = np.sum(delta, axis=0) / batch_size
        for i in xrange(2, self.num_layers):
            delta = delta.dot(self.w[-i+1].T) * self.sigmoidPrime(self.z[-i])
            self.grad_w[-i] = ((delta.T).dot(self.a[-i - 1])) / batch_size
            self.grad_b[-i] = np.sum(delta, axis=0) / batch_size

        # regularization
        for i in xrange(self.num_layers-1):
            self.grad_w[i] = self.grad_w[i].T + (self.reg_lambda * self.w[i]) / self.training_size

    def gradientDescent(self, X, y, regularization, learning_rate, epochs, output=False):
        self.training_size = float(len(X))

        for e in xrange(epochs):
            self.feedForward(X)
            self.backpropagation(y)

            # update weights
            for l in range(self.num_layers-1):
                self.w[l] = self.w[l] - learning_rate * self.grad_w[l]
                self.b[l] = self.b[l] - learning_rate * self.grad_b[l]
            
            if output:
                predictions = np.argmax(self.a[-1], axis=1)
                precision = self.evaluatePredictions(predictions, np.nonzero(y)[1])
                print "Epoch: {0} - precision: {1:.4f}, cost: {2:.4f}".format(e, precision, self.cost(y))


    def stochasticGradientDescent(self, X, y, regularization, learning_rate, epochs, batch_size, output=False):
        self.training_size = float(len(X))
        for e in xrange(epochs):
            X, y = self.shuffleData(X, y)
            batches = [(X[i:i+batch_size], y[i:i+batch_size]) for i in xrange(0, int(self.training_size), batch_size)]
            for batch in batches:
                Xi = batch[0]
                yi = batch[1]

                self.feedForward(Xi)
                self.backpropagation(yi)

                # update weights
                for l in xrange(self.num_layers-1):
                    self.w[l] = self.w[l] - learning_rate * self.grad_w[l]
                    self.b[l] = self.b[l] - learning_rate * self.grad_b[l]
            
            if output:
                self.outputTrainingStatus(e, X, y)

    def outputTrainingStatus(self, epoch_num, X, y):
        y1 = np.nonzero(y)[1]
        predictions = self.predict(X)
        precision = self.evaluatePredictions(predictions, y1)
        print "Epoch: {0} - precision: {1:.4f}, cost: {2:.4f}".format(epoch_num, precision, self.cost(y))

    def shuffleData(self, X, y):
        c = list(zip(X, y))
        random.shuffle(c)
        X, y = zip(*c)
        return (np.asarray(X), np.asarray(y))

    def predict(self, X):
        self.feedForward(X)
        return np.argmax(self.a[-1], axis=1)

    def evaluatePredictions(self, predictions, y):
        return (predictions == y).sum() / float(len(y))

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoidPrime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
