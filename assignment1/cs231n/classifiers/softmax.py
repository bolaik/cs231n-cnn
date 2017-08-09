import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  # Compute the softmax loss and its gradient using explicit loops. 
  # Store the loss in loss and the gradient in dW. If you are not careful 
  # here, it is easy to run into numeric instability. Don't forget the   
  # regularization!                                                     
  
  # shape
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in range(num_train):  
    f = X[i].dot(W)
    f -= np.max(f)
    f_correct = f[y[i]]

    # loss and gradient
    sum_i = 0.0
    for j in range(num_classes):
      sum_i += np.exp(f[j])
    loss += -f_correct + np.log(sum_i)

    for j in range(num_classes):
      p = np.exp(f[j]) / sum_i
      dW[:,j] += (p - (j==y[i])) * X[i,:].T

    # average
  loss /= num_train
  dW /= num_train

    # regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  # Compute the softmax loss and its gradient using no explicit loops. 
  # Store the loss in loss and the gradient in dW. If you are not careful
  # here, it is easy to run into numeric instability. Don't forget the  
  # regularization!                                                  
  
  # shape
  num_classes = W.shape[1]
  num_train = X.shape[0]

  # scores matrix
  f = X.dot(W)
  # numeric stability
  f -= np.array([np.max(f,1)]).T

  # loss
  f_correct = f[range(num_train), y]
  loss = np.mean(-f_correct + np.log(np.sum(np.exp(f),1)))
  # regularization
  loss += 0.5 * reg * np.sum(W * W)

  # softmax probability matrix
  p = np.exp(f) / np.array([np.sum(np.exp(f),1)]).T
  p[range(num_train), y] -= 1
  # gradient
  dW = X.T.dot(p)
  dW /= num_train
  # regularization
  dW += reg * W

  return loss, dW
