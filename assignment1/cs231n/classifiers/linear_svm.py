import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

        # compute gradients,reference to notes
        dW[:,y[i]] -= X[i,:].T  # for y_i, perform sum for j != y_i
        dW[:,j] += X[i,:].T     # for other j
      

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  dW /= num_train # average over train size

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  dW += reg * W # regularization

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # Implement a vectorized version of the structured SVM loss    
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)

  # the extremely tricky part is subset the correct_score from scores
  # fast index in both directions scores[np.arange(num_train), y]
  # then transform to 2d array and transpose
  margins = np.maximum(0, scores - np.array([scores[np.arange(num_train),y]]).T + 1)
  margins[np.arange(num_train),y] = 0
  loss = np.sum(margins) / num_train

  loss += 0.5 * reg * np.sum(W * W)
  
  # Implement a vectorized version of the gradient for the structured SVM loss.
  # Hint: Instead of computing the gradient from scratch, it may be easier to 
  # reuse some of the intermediate values that you used to compute the loss.
  binary = margins
  binary[margins > 0] = 1    # L_{i,y_j} > 0
  binary[range(num_train), y] = -np.sum(binary, axis=1)
  dW = X.T.dot(binary)

  # normalize
  dW /= num_train

  # regularize
  dW += reg * W
  
  return loss, dW
