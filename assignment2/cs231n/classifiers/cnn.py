from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """

        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # Initialize weights and biases for the three-layer convolutional network. #
        # Weights should be initialized from a Gaussian with standard deviation    #
        # equal to weight_scale; biases should be initialized to zero. All weights #
        # and biases should be stored in the dictionary self.params. Store weights #
        # and biases for the convolutional layer using the keys 'W1' and 'b1'; use #
        # keys 'W2' and 'b2' for the weights and biases of the hidden affine       #
        # layer, and keys 'W3' and 'b3' for the weights and biases of the output   #
        # affine layer.                                                            #
        ############################################################################

        # layer 1: conv layer
        # (N, C, H, W) -> (N, F, Hc, Wc)
        C, H, W = input_dim
        F = num_filters
        HH, WW = filter_size, filter_size
        P, S = (HH - 1) // 2, 1
        Hc = 1 + (H + 2 * P - HH) // S
        Wc = 1 + (W + 2 * P - WW) // S

        W1 = weight_scale * np.random.randn(F, C, HH, WW)
        b1 = np.zeros(F)

        # layer 2: max pooling layer
        # (N, F, Hc, Wc) -> (N, F, Hp, Wp)
        pH, pW, pS = 2, 2, 2
        Hp = 1 + (Hc - pH) // pS
        Wp = 1 + (Wc - pW) // pS

        # layer 3: hidden affine layer
        # (F, Hp, Wp) -> (hidden_dim,)
        W2 = weight_scale * np.random.randn(F * Hp * Wp, hidden_dim)
        b2 = np.zeros(hidden_dim)

        # layer 4: output affine layer
        # (hidden_dim,) -> (num_classes,)
        W3 = weight_scale * np.random.randn(hidden_dim, num_classes)
        b3 = np.zeros(num_classes)

        self.params.update({'W1': W1, 'b1': b1,
                            'W2': W2, 'b2': b2,
                            'W3': W3, 'b3': b3})

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # Implement the forward pass for the three-layer convolutional net,        #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        N, C, H, W = X.shape
        out_conv, cache_conv = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out_fc1, cache_fc1 = affine_relu_forward(out_conv.reshape(N,-1), W2, b2)
        out_fc2, cache_fc2 = affine_forward(out_fc1, W3, b3)
        scores = out_fc2

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # Implement the backward pass for the three-layer convolutional net,       #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################

        # compute softmax loss and gradient
        loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))
        loss += reg_loss

        # back propagate affine layer
        dout_fc1, dW3, db3 = affine_backward(dscores, cache_fc2)
        dW3 += self.reg * W3

        # back propagate affine_relu layer
        dout_conv, dW2, db2 = affine_relu_backward(dout_fc1, cache_fc1)
        dW2 += self.reg * W2
        dout_conv = dout_conv.reshape(out_conv.shape)

        # back propagate conv_relu_pool layer
        dX, dW1, db1 = conv_relu_pool_backward(dout_conv, cache_conv)
        dW1 += self.reg * W1

        grads.update({'W1': dW1, 'b1': db1,
                      'W2': dW2, 'b2': db2,
                      'W3': dW3, 'b3': db3})

        return loss, grads
