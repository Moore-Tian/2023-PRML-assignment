"""
Activation functions

ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py
"""

import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from fdunn.modules.base import Module


class Sigmoid(Module):
    """Applies the element-wise function:
    .. math::
    \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}

    Shape:
    - input: :math:`(*)`, where :math:`*` means any number of dimensions.
    - output: :math:`(*)`, same shape as the input.
    """
    def __init__(self):
        self.input = None
        self.output = None
        self.params = None

    def forward(self, input):
        ###########################################################################
        # TODO:                                                                   #
        # Implement the forward method.                                           #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        output = 1 / (1 + np.exp(-input))
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.output = output
        return output

    def backward(self, output_grad):
        """
        Input:
            - output_grad:(*)
            partial (loss function) / partial (output of this module)

        Return:
            - input_grad:(*)
            partial (loss function) / partial (input of this module)
        """
        ###########################################################################
        # TODO:                                                                   #
        # Implement the backward method.                                          #
        ###########################################################################
        input_grad = output_grad * ((1 - self.output) * self.output)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return input_grad
    

# 一个ReLU激活层的实现
class ReLU(Module):
    def __init__(self):
        self.input = None
        self.output = None
        self.params = None

    def forward(self, input):
        self.input = input
        output = np.where(input > 0, input, 0)
        return output
    
    def backward(self, output_grad):
        input_grad = np.where(self.input > 0, output_grad, 0)
        return input_grad