import math

def sigmoid(h):
  # activation function
  y = 1.0/(1.0 + math.exp(-h))
  return y

def activate(inputs, weights):
  # perform net input
  h = 0
  for x, w in zip(inputs, weights):
    h += x*w
    
  # perform activation
  return sigmoid(h)

if __name__ == "__main__":
  
  #         x1  x2  x3
  inputs = [.5, .3, .2]
  
  #          w1  w2  w3
  weights = [.4, .7, .2]
  
  output = activate(inputs, weights)
  print(output)