# Neural Network from scratch using Python 
This repository implements a neural network using pure python (and Numpy for matrix operations).
The objective of this project is put into practice all the process of building and training a neural network, implementing the feedforward, backpropagation, gradient descent, and stochastic gradient descent. This implementation is tested with the MNIST dataset for digits recognition, achieving 96% of precision.

## More information
For more information about the implementation and how to use it, you can check the comments along the code or, for a detailed tutorial, access: http://www.rafaelglater.com/

## About the files
Three files worth noting:
* simple_example.py
* neural_network.py
* main.py

### simple_example.py
It's a stand-alone script which implements a shallow network to recognize digits using the MNIST dataset. This code aims to be very simple and didactic. To keep as simple as possible, I hardcoded the network and its parameters specifically for this example. The idea here is to get insights on how to code each step of a neural network.

### neural_network.py
A generic class implementation of a neural network. With this class, you can create a neural network specifying the number of
layers and neurons in each one.

### main.py
A usage example of NeuralNetwork class to recognize digits using the MNIST dataset.


## MNIST Dataset
The MNIST database contains 70,000 examples of handwritten digits, divided into 60,000 examples for training and 10,000 to test.  The digits have been size-normalized and centered in a fixed-size image. For more information about this dataset and download, access: http://yann.lecun.com/exdb/mnist/
