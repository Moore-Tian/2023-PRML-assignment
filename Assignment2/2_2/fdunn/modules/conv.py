"""
Conv2D

ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from fdunn.modules.base import Module
from .utils import *




class Conv2d(Module):
    """Applies a 2D convolution over an input signal composed of several input
    planes.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of
            the input. Default: 0
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})` or :math:`(C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` or :math:`(C_{out}, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
            :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``,
            then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride = 1,
            padding = 1,
            bias = True
    ):
        # input and output
        self.input = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.X_col = None
        self.W_col = None

        # params
        self.params = {}
        ###########################################################################
        # TODO:                                                                   #
        # Implement the params init.                                              #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.params['W'] = None
        self.params['b'] = None

        std = np.sqrt(2 / (self.in_channels + self.out_channels))

        self.params['W'] = np.random.normal(0, std, size=(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]))

        if bias:
            self.params['b'] = np.zeros((self.out_channels, ))
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # grads of params
        self.grads = {}

    def my_conv(self, X, Kernel, stride=1, einsum_formula='bcd,abcd->a', out_ele_shape=None):
        rows, cols = X.shape[-2:]
        kernel_rows, kernel_cols = Kernel.shape[-2:]
        out_shape = (*out_ele_shape, (rows - kernel_rows)//stride + 1, (cols - kernel_cols)//stride + 1)
        out = np.zeros(out_shape)
        slice_base_idx = (slice(None),) * (len(out.shape) - 2)
        for i in range(out_shape[-2]):
            for j in range(out_shape[-1]):
                X_window = X[slice_base_idx + (slice(i*stride, i*stride+kernel_rows), slice(j*stride, j*stride+kernel_cols))]
                out[slice_base_idx + (i, j)] = np.einsum(einsum_formula, X_window, Kernel)
        return out

    def forward(self, input):
        if isinstance(input, np.ndarray) is False:
            input = input.numpy()
        self.input = input
        ###########################################################################
        # TODO:                                                                   #
        # Implement the forward method.                                           #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        batchsize, _, in_H, in_W = input.shape

        out_H = (in_H - self.kernel_size[0] + 2 * self.padding) // self.stride + 1
        out_W = (in_W - self.kernel_size[1] + 2 * self.padding) // self.stride + 1

        # 这里使用了快速卷积，利用im2col将卷积运算转换为了矩阵乘法，大大加快了运算速度
        input_col = im2col(input, self.kernel_size[0], self.kernel_size[1], self.stride, self.padding)
        W_col = self.params['W'].reshape((self.out_channels, -1))
        output = np.dot(W_col, input_col)
        output = np.array(np.hsplit(output, batchsize)).reshape((batchsize, self.out_channels, out_H, out_W))
        self.W_col = W_col
        self.input_col = input_col

        if self.params['b'] is not None:
            output += self.params['b'][:, np.newaxis, np.newaxis]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return output

    def backward(self, output_grad):
        ###########################################################################
        # TODO:                                                                   #
        # Implement the backward method.                                          #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        batch_size = output_grad.shape[0]
        output_grad_col = output_grad.reshape((output_grad.shape[0] * output_grad.shape[1], -1))
        output_grad_col = np.array(np.vsplit(output_grad_col, batch_size))
        output_grad_col = np.concatenate(output_grad_col, axis=-1)

        # 同样用矩阵乘法的视角计算梯度，加快计算速度
        self.grads['W'] = np.dot(output_grad_col, self.input_col.T).reshape(self.params['W'].shape)
        self.grads['b'] = np.mean(output_grad, axis=(0, 2, 3))
        input_grad = np.dot(self.W_col.T, output_grad_col)
        input_grad = col2im(input_grad, self.input.shape, self.kernel_size[0], self.kernel_size[1], self.stride, self.padding)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return input_grad
    
    
