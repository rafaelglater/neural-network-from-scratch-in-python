'''
This is a simple implementation of a neural network with just one hidden layer
to deal with digit recognition using the MNIST dataset.
Download the dataset (http://yann.lecun.com/exdb/mnist/) and copy the
unzipped files to the "mnist_digits_dataset" folder.

This example aims to be entirely didactic and, because of that, the 
implementation is hardcoded for this dataset. The idea is to give an intuition
of how to code each step of a neural network in a straightforward manner, 
giving you the ability to generalize later for different situations and sizes 
of neural networks.

To a more general neural network implementation, have a look in the 
"neural_network.py" file for a generic class implementation.
'''

from mnist import MNIST
import numpy as np
import random


def main():
    input_layer_size = 784  # 28x28 image
    hidden_layer_size = 30
    output_layer_size = 10
    reg_lambda = 5.0  # regularization
    lr = 0.1  # learning rate
    batch_size = 10
    epochs = 30

    # initialize weights and biases from a uniform distribution varying around 0
    # we can also initialize using a standart normal distribution: np.random.randn
    w1 = np.random.uniform(-0.12, 0.12, (input_layer_size, hidden_layer_size))
    w2 = np.random.uniform(-0.12, 0.12, (hidden_layer_size, output_layer_size))
    b1 = np.random.uniform(-0.12, 0.12, (1, hidden_layer_size))
    b2 = np.random.uniform(-0.12, 0.12, (1, output_layer_size))

    # load the dataset using a third party library
    mndata = MNIST('./mnist_digits_dataset')
    images_train, labels_train = mndata.load_training()
    images_test, labels_test = mndata.load_testing()
    training_size = float(len(images_train))

    # we can view the digits using the command bellow:
    # print mndata.display(images_train[0])

    # convert to numpy arrays to make our life easier
    images_train = np.asarray(images_train)
    images_test = np.asarray(images_test)
    labels_test = np.asarray(labels_test)

    # normalize to the [0,1] scale (this brings better results)
    images_train = images_train / 255.0
    images_test = images_test / 255.0

    X = images_train
    y = labels_train

    # transform y in a vector of zeros and one.
    # I.e. number "5" will be = [0 0 0 0 1 0 0 0 0 0]
    y = oneHotEncode(y, output_layer_size)

    # Here we start the stochastic gradient descent. In each epoch, we shuffle
    # the data, break into small batches and calculate the gradients
    for e in xrange(epochs):
        X, y = shuffleData(X, y)
        batches = [(X[i:i + batch_size], y[i:i + batch_size])
                   for i in xrange(0, int(training_size), batch_size)]
        for batch in batches:
            Xi = batch[0]
            yi = batch[1]

            # STEP 1: feedforward --------------------------------------------
            a1 = Xi
            z2 = a1.dot(w1) + b1
            a2 = sigmoid(z2)
            z3 = a2.dot(w2) + b2
            a3 = sigmoid(z3)

            # STEP 2: backpropagation ----------------------------------------
            delta3 = (a3 - yi)
            grad_w2 = (delta3.T).dot(a2) / batch_size
            grad_b2 = np.sum(delta3, axis=0) / batch_size

            delta2 = delta3.dot(w2.T) * sigmoidPrime(z2)
            grad_w1 = (delta2.T).dot(a1) / batch_size
            grad_b1 = np.sum(delta2, axis=0) / batch_size

            # regularization of backpropagation
            grad_w2 = grad_w2.T + (reg_lambda * w2) / training_size
            grad_w1 = grad_w1.T + (reg_lambda * w1) / training_size

            # STEP 3: update weights -----------------------------------------
            w1 = w1 - lr * grad_w1
            b1 = b1 - lr * grad_b1
            w2 = w2 - lr * grad_w2
            b2 = b2 - lr * grad_b2

        # Let's check the cost in this epoch
        print "Epoch: {0} - cost: {1}".format(e, cost(X, y, w1, b1, w2, b2, reg_lambda))

    # our neural network is already trained. Now we'll check its precision on
    # identifying digits in the test set
    predictions = predict(images_test, w1, b1, w2, b2)
    print "Precision in the test set: {0}".format(evaluatePredictions(predictions, labels_test))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoidPrime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def predict(X, w1, b1, w2, b2):
    a1 = X
    z2 = a1.dot(w1) + b1
    a2 = sigmoid(z2)
    z3 = a2.dot(w2) + b2
    a3 = sigmoid(z3)
    return np.argmax(a3, axis=1)


def evaluatePredictions(predictions, y):
    return (predictions == y).sum() / float(len(y))


def shuffleData(X, y):
    c = list(zip(X, y))
    random.shuffle(c)
    X, y = zip(*c)
    return (np.asarray(X), np.asarray(y))


def cost(X, y, w1, b1, w2, b2, reg_lambda):
    a1 = X
    z2 = a1.dot(w1) + b1
    a2 = sigmoid(z2)
    z3 = a2.dot(w2) + b2
    a3 = sigmoid(z3)

    sample_size = len(X)

    # comput the difference between each output unit K and unit in y
    diff_Ks = -y * np.log(a3) - (1 - y) * np.log(1 - a3)
    # first sum the Ks, then sum all samples
    J = sum(np.sum(diff_Ks, axis=0)) / sample_size
    
    # regularization part
    sum1 = sum(w1 ** 2)
    sum2 = sum(w2 ** 2)
    sum3 = sum(np.concatenate((sum1, sum2)))
    J = J + ((reg_lambda * sum3) / (2 * sample_size))
    return J


def oneHotEncode(y, output_layer_size):
    num_samples = len(y)

    y1 = np.zeros((num_samples, output_layer_size))
    for i in xrange(num_samples):
        y1[i, :] = [item == y[i] for item in xrange(output_layer_size)]
    return y1



if __name__ == '__main__':
	main()
