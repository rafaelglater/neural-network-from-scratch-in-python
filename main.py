'''
This is an example of using the NeuralNetwork class to recognize digits using
the MNIST dataset. Download the dataset (http://yann.lecun.com/exdb/mnist/) 
and copy the unzipped files to the "mnist_digits_dataset" folder.

For a simpler and more didactic neural network implementation, please check the
"simple_example.py" file.
'''

import time
import numpy as np
from mnist import MNIST
from neural_network import NeuralNetwork


# load the dataset
mndata = MNIST('./mnist_digits_dataset')
images_train, labels_train = mndata.load_training()
images_test, labels_test = mndata.load_testing()

# convert to numpy arrays
images_train = np.asarray(images_train)
images_test = np.asarray(images_test)
labels_test = np.asarray(labels_test)

# normalize to [0,1] scale
images_train = images_train / 255.0
images_test = images_test / 255.0

X = images_train
y = labels_train

# transform y in a vector of zeros and one. 
# I.e. number "5" will be = [0 0 0 0 0 1 0 0 0 0]
output_layer_size = 10
m = len(images_train)
y1 = np.zeros((m, output_layer_size))
for i in range(m):
    y1[i, :] = [item == y[i] for item in range(output_layer_size)]
y = y1


start = time.time()

# let's create our neural network (with one hidden layer) and train it using SGD
nn = NeuralNetwork(784, 30, 10)
nn.stochasticGradientDescent(X, y, regularization=5.0, learning_rate=0.1, epochs=30, batch_size=10, output=True)

# you can also use gradient descent
# nn.gradientDescent(X, y, regularization=1.0, learning_rate=0.3, epochs=150, output=True)

# our neural network is already trained. Now we'll check its precision on
# identifying digits in the test set
predictions = nn.predict(images_test)
print "Precision in test: {0}".format(nn.evaluatePredictions(predictions, labels_test))

print 'Time in seconds: ' + str(time.time() - start)

