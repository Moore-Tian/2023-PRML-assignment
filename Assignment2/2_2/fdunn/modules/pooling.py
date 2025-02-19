"""
Pooling

ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/pooling.py
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from fdunn.modules.base import Module
from .utils import *


class MaxPool2d(Module):
    """Applies a 2D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
        \end{aligned}

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                    \times (\text{kernel\_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                    \times (\text{kernel\_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor
    """
    def __init__(
            self,
            kernel_size,
            stride
        ):
        # input and output
        self.input = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.params = None
        self.cache = None

    def forward(self, input):
        self.input = input
        ###########################################################################
        # TODO:                                                                   #
        # Implement the forward method.                                           #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        batch_size, channel, in_H, in_W = input.shape
        out_H = (in_H - self.kernel_size[0]) // self.stride + 1
        out_W = (in_W - self.kernel_size[1]) // self.stride + 1

        indices = pool2row_indices(input, self.kernel_size[0], self.kernel_size[1], self.stride)
        output = np.max(indices, axis=1)

        arg_output = np.argmax(indices, axis=1)
        self.cache = (arg_output, input.shape, indices.shape)
        output = pool_fc2output(output, batch_size, out_H, out_W)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return output

    def backward(self, output_grad):
        ###########################################################################
        # TODO:                                                                   #
        # Implement the backward method.                                          #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        arg_output, input_shape, indices_shape = self.cache

        output_grad = pool_output2fc(output_grad)
        indices_grad = np.zeros(indices_shape)
        indices_grad[range(indices_shape[0]), arg_output] = output_grad

        input_grad = row2pool_indices(indices_grad, input_shape, self.kernel_size[0], self.kernel_size[1], self.stride)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return input_grad