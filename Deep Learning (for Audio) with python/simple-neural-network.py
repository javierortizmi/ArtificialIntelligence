import numpy as np
from random import random

# save activations derivatives
# implement backpropagation
# implement gradient descent
# implement train
# train our net with some dummy dataset
#* make some predictions

# MLP stands for Multi-Layer Perceptron
class MLP:
  
  def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2):
    
    self.num_inputs = num_inputs
    self.num_hidden = num_hidden
    self.num_outputs = num_outputs
    
    # create a generic representation of the layers
    layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]
    
    # create random connection weights for the layers
    weights = []
    for i in range(len(layers)-1):
      w = np.random.rand(layers[i], layers[i+1])
      weights.append(w)
    self.weights = weights
    
    activations = []
    for i in range(len(layers)):
      a = np.zeros(layers[i])
      activations.append(a)
    self.activations = activations
    
    derivatives = []
    for i in range(len(layers)-1):
      d = np.zeros((layers[i], layers[i+1]))
      derivatives.append(d)
    self.derivatives = derivatives
      
  def forward_propagate(self, inputs):
    
    # the input layer activation is just the input itself
    activations = inputs
    self.activations[0] = inputs
    
    # iterate through the network layers
    for i, w in enumerate(self.weights):
      
      # calculate net inputs
      net_inputs = np.dot(activations, w)
      
      # calculate the activations
      activations = self._sigmoid(net_inputs)
      self.activations[i+1] = activations
      
    # return output layer activation
    return activations 
  
  def back_propagate(self, error, verbose=False):
    # dE/dW_i = (y - a_[i+1]) s'(h_[i+1]) a_i
    # s'(h_[i+1]) = s(h_[i+1])(1 - s(h_[i+1]))
    # s(h_[i+1]) = a_[i+1]
    
    # dE/dW_[i-1] = (y - a_[i+1]) s'(h_[i+1]) W_i s'(h_i) a_[i-1]
    
    # we need to calculate dE/dW_i for every weight in the network
    # dE/dW_i = (y - a_[i+1]) s'(h_[i+1]) a_i
    for i in reversed(range(len(self.derivatives))):
      
      activations = self.activations[i+1]
      delta = error * self._sigmoid_derivative(activations)   # ndarray([0.1, 0.2]) --> ndarray([[0.1, 0.2]])
      delta_reshaped = delta.reshape(delta.shape[0], -1).T
      
      current_activations = self.activations[i]   # ndarray([0.1, 0.2]) --> ndarray([[0.1], [0.2]])
      current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
      
      self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
      
      error = np.dot(delta, self.weights[i].T)
      
      if verbose:
        print("Derivatives for W{}:\n {}".format(i, self.derivatives[i]))
      
    return error
  
  def gradient_descent(self, learning_rate, verbose=False):
    for i in range(len(self.weights)):
      weights = self.weights[i]
      if verbose:
        print("Original W{}:\n {}".format(i, weights))
      derivatives = self.derivatives[i]
      weights += derivatives * learning_rate
      if verbose:
        print("Updated W{}:\n {}".format(i, weights))
        
  def train(self, inputs, targets, epochs, learning_rate):
    # epochs is the number of times to loop through the entire dataset
    for i in range(epochs):
      sum_errors = 0
      
      for input, target in zip(inputs, targets):
        # perform forward propagation
        output = self.forward_propagate(input)
        
        # calculate the error
        error = target - output
        
        # perform backpropagation
        self.back_propagate(error)
        
        # now perform gradient descent on the derivatives
        # (this will update the weights)
        self.gradient_descent(learning_rate)
        
        # keep track of the MSE(Mean Squared Error) for reporting later
        sum_errors += self._mse(target, output)
        
      # Epoch complete, report the training error
      print("Error: {} at epoch {}".format(sum_errors / len(inputs), i+1))
      
    print("Training complete!")
    print("=====")
    
  def _mse(self, target, output):
    return np.average((target - output) ** 2)
    
  def _sigmoid_derivative(self, x):
    return x * (1.0 - x) 
  
  def _sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
    
if __name__ == "__main__":
  
  # create an MLP
  mlp = MLP(2, [5], 1)
  
  # create a dataset to train the mlp for the sum operation
  inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)]) # array([[0.1, 0.2], [0.3, 0.4], ...])
  targets = np.array([[i[0] + i[1]] for i in inputs])                        # array([[0.3], [0.7], ...])
  
  # train our mlp
  mlp.train(inputs, targets, 50, 0.1)
  
  # create dummy data
  input = np.array([0.1, 0.3])
  target = np.array([0.4])
  
  output = mlp.forward_propagate(input)
  
  print("\nOur network believes that {} + {} is equal to {}\n".format(input[0], input[1], output[0]))