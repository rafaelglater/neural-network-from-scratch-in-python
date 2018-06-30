'''
You can ignore this file! If you are looking for a simple example of neural
network implementation, please have a look on the "simple_example.py" file.

Or if you want a general class implementation, check the "neural_network.py".

This draft code is just an attempt to "attach" the bias direct to the matrix 
of weights, instead of sum up the biases later. But the performance wasn't so 
good as to sum separately (probably because of the operations of adding ones 
and zeros to the matrix).
But I decided to keep this code for future reference.
'''


from mnist import MNIST
import numpy as np
import pandas as pd
import math

def sigmoid(z):
  return 1.0 / (1.0 + np.exp(-z))

def sigmoidGradient(z):
  return sigmoid(z) * (1 - sigmoid(z))

def addOnesColumn(m):
  aux = np.ones((m.shape[0], m.shape[1]+1))
  aux[:, 1:] = m
  return aux

def addZerosColumn(m):
  aux = np.zeros((m.shape[0], m.shape[1]+1))
  aux[:, 1:] = m
  return aux

def predict(X, theta1, theta2):
  X = addOnesColumn(X)
  r1 = sigmoid(X.dot(theta1.transpose()))
  r1 = addOnesColumn(r1)                             # results from first layer
  h = sigmoid(r1.dot(theta2.transpose()))            # results from second layer (output layer)

  return np.argmax(h, axis=1)

def evaluate_predictions(predictions, y):
  return (predictions == y).sum() / float(len(y))


def costFunction(theta1, theta2, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa):
  m = len(X)

  # add column of ones
  X1 = np.ones((m, input_layer_size+1))
  X1[:,1:] = X

  # transform y in a vector of zeros and one. I.e. number "5" will be = [0 0 0 0 1 0 0 0 0 0]
  y1 = np.zeros((m, num_labels))
  for i in range(m):
    y1[i, :] = [item == y[i] for item in range(num_labels)]
  y = y1
  
  # FEEDFORWARD
  r1 = sigmoid(X1.dot(theta1.transpose()))
  r1 = addOnesColumn(r1)                             # results from first layer
  h = sigmoid(r1.dot(theta2.transpose()))            # results from second layer (output layer)
  

  # COST
  diff_Ks = -y * np.log(h) - (1 - y) * np.log(1 - h) # comput the difference between each output unit K and unit in y
  J = sum(np.sum(diff_Ks, axis=0)) / m              # first sum the Ks, then sum all samples

  # regularization part
  sum1 = sum(theta1[:,1:] ** 2)
  sum2 = sum(theta2[:,1:] ** 2)
  sum3 = sum(np.concatenate((sum1, sum2)))
  J = J + ( (lambdaa * sum3) / (2 * m) )
  
  
  # BACKPROPAGATION
  a1 = X1
  z2 = a1.dot(theta1.transpose())
  a2 = sigmoid(z2)
  a2 = addOnesColumn(a2)
  z3 = a2.dot(theta2.transpose())
  a3 = sigmoid(z3)

  delta3 = a3 - y
  delta2 = delta3.dot(theta2[:,1:]) * sigmoidGradient(z2)

  theta1_grad = delta2.transpose().dot(a1) / m
  theta2_grad = delta3.transpose().dot(a2) / m

  # regularization
  theta1_grad = theta1_grad + ((lambdaa * addZerosColumn(theta1[:,1:]))/m)
  theta2_grad = theta2_grad + ((lambdaa * addZerosColumn(theta2[:,1:]))/m)

  return (J, theta1_grad, theta2_grad)

################################


mndata = MNIST('./mnist_digits_dataset')

images_train, labels_train = mndata.load_training()
images_test, labels_test = mndata.load_testing()

# we can view the digits using the command bellow:
# print mndata.display(images_train[0])

# convert to numpy arrays to make our life easier
images_train = np.asarray(images_train)
images_test = np.asarray(images_test)
labels_test = np.asarray(labels_test)

# normalize to the [0,1] scale (this brings better results)
images_train = images_train / 255.0
images_test = images_test / 255.0


input_layer_size = 784 # 28x28 image
hidden_layer_size = 50
num_labels = 10
lambdaa = 1 # regularization
m = len(images_train)


# initialize thetas randomly
theta1 = np.random.uniform(-0.12, 0.12, (hidden_layer_size, input_layer_size+1))
theta2 = np.random.uniform(-0.12, 0.12, (num_labels, hidden_layer_size+1))

learning_rate = 0.1
(J, theta1_grad, theta2_grad) = costFunction(theta1, theta2, input_layer_size, hidden_layer_size, num_labels, images_train, labels_train, lambdaa)
print J
for i in range(50):
  theta1 = theta1 - learning_rate * theta1_grad
  theta2 = theta2 - learning_rate * theta2_grad
  (J, theta1_grad, theta2_grad) = costFunction(theta1, theta2, input_layer_size, hidden_layer_size, num_labels, images_train, labels_train, lambdaa)

print J

predictions = predict(images_test, theta1, theta2)

print evaluate_predictions(predictions, labels_test)

