# -*- coding: utf-8 -*-
# @Author  : Ehwartz
# @Github  : https://github.com/Ehwartz
# @Time    : 05/21/2024
# @Software: PyCharm
# @File    : nn.py

import numpy as np
from tensor import Tensor
from utils import fold, unfold, fold_grad
import inspect


class module(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.parameters = []

    def forward(self, x: Tensor):
        self.x = x
        x.im = self

        y = x

        self.y = y
        y.om = self
        return y

    def backward(self):
        pass

    def __call__(self, x: Tensor):
        return self.forward(x)


class Linear(module):
    def __init__(self, in_features, out_features, bias):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w = Tensor(shape=[in_features, out_features])
        self.b = Tensor(shape=[out_features])
        self.if_bias = bias
        self.parameters = [self.w, self.b]

    def forward(self, x: Tensor):
        self.x = x
        x.im = self
        n = x.data.shape[0]
        y = Tensor([n, self.out_features])
        y.data = np.matmul(x.data, self.w.data)
        self.y = y
        y.om = self
        return y

    def backward(self):
        self.w.grad += np.matmul(self.x.data.reshape([-1, self.in_features, 1]),
                                 self.y.grad.reshape([-1, 1, self.out_features])).sum(axis=0)
        self.x.grad = np.matmul(self.y.grad, self.w.data.transpose([1, 0]))
        self.x.backward()


class Conv2D(module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2D, self).__init__()
        self.ic = in_channels
        self.oc = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel = Tensor([self.ic, self.oc, self.kernel_size, self.kernel_size])
        self.unfolded_kernel_size = [self.ic * self.kernel_size * self.kernel_size, self.oc]
        self.unfolded_kernel = self.kernel.transpose([0, 2, 3, 1]).reshape(self.unfolded_kernel_size)
        self.unfolded_x = None
        self.unfolded_y = None
        self.unfolded_x_shape = None
        self.unfolded_y_shape = None
        self.parameters = [self.unfolded_kernel]

    def forward(self, x: Tensor):
        self.x = x
        x.im = self
        n, xc, xh, xw = x.shape
        yc = self.oc
        yh = (xh + self.padding * 2 - self.kernel_size) // self.stride + 1
        yw = (xw + self.padding * 2 - self.kernel_size) // self.stride + 1
        unfolded_x_data = unfold(x.data, self.kernel_size, self.stride, self.padding)
        self.unfolded_x = Tensor(unfolded_x_data.shape)
        self.unfolded_x.data = unfolded_x_data
        self.unfolded_y_shape = [n, yh * yw, yc]
        self.unfolded_y = Tensor(self.unfolded_y_shape)
        self.unfolded_y.data = np.matmul(unfolded_x_data, self.unfolded_kernel.data) / self.ic
        y = self.unfolded_y.reshape([n, yc, yh, yw])
        self.y = y
        y.om = self
        return y

    def backward(self):
        n, yc, yh, yw = self.y.data.shape
        unfolded_y_grad = self.y.grad.reshape([n, yh * yw, yc])
        self.unfolded_kernel.grad += np.matmul(self.unfolded_x.data.reshape([n, yh * yw,
                                                                             self.ic *
                                                                             self.kernel_size * self.kernel_size,
                                                                             1]),
                                               unfolded_y_grad.reshape([n, yh * yw, 1, yc])
                                               ).sum(axis=0).sum(axis=0) / self.ic
        unfolded_x_grad = np.matmul(unfolded_y_grad, self.unfolded_kernel.data.transpose([1, 0]))
        self.x.grad = fold_grad(unfolded_x_grad, self.x.shape, self.kernel_size, self.stride, self.padding) / self.ic
        self.x.backward()


class ReLU(module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.mask = np.array([])

    def forward(self, x: Tensor):
        self.x = x
        x.im = self

        y = Tensor(shape=self.x.shape)
        self.mask = 1 * (x.data >= 0)
        y.data = x.data * self.mask
        self.y = y
        y.om = self
        return y

    def backward(self):
        self.x.grad = self.y.grad * self.mask
        self.x.backward()


class LeakyReLU(module):
    def __init__(self, alpha):
        super(LeakyReLU, self).__init__()
        self.alpha = alpha
        self.mask = np.array([])

    def forward(self, x: Tensor):
        self.x = x
        x.im = self

        y = Tensor(shape=self.x.shape)
        self.mask = 1 * (x.data >= 0) + self.alpha * (x.data < 0)
        y.data = x.data * self.mask
        self.y = y
        y.om = self
        return y

    def backward(self):
        self.x.grad = self.y.grad * self.mask
        self.x.backward()


class Reshape(module):
    def __init__(self, x_shape, y_shape):
        super(Reshape, self).__init__()
        self.x_shape = x_shape
        self.y_shape = y_shape

    def forward(self, x: Tensor):
        self.x = x
        x.im = self

        y_data = x.data.reshape(self.y_shape)
        y_grad = x.grad.reshape(self.y_shape)
        y = Tensor(shape=y_data.shape)
        y.data = y_data
        y.grad = y_grad
        self.y = y
        y.om = self
        return y

    def backward(self):
        self.x.grad = self.y.grad.reshape(self.x_shape)
        self.x.backward()


class loss(object):
    def __init__(self):
        self.x = None
        self.t = None
        self.y = None

    def forward(self, x: Tensor, t: Tensor):
        return

    def __call__(self, x: Tensor, t: Tensor):
        return self.forward(x, t)


class MSELoss(loss):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, x: Tensor, t: Tensor):
        self.x = x
        x.im = self
        self.t = t

        y_data = np.square(self.x.data - self.t.data).sum()
        y = Tensor([1])
        y.data = y_data
        self.y = y
        y.om = self
        return y

    def backward(self):
        self.x.grad = 2 * (self.x.data - self.t.data)
        self.x.backward()

    def __call__(self, x: Tensor, t: Tensor):
        return self.forward(x, t)


class CrossEntropyLoss(loss):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.x = None
        self.y = None
        self.p = None
        self.parameters = []

    def forward(self, x: Tensor, t: Tensor):
        self.x = x
        x.im = self
        self.t = t
        x_exp = np.exp(x.data)
        x_exp_sum = np.expand_dims(np.sum(x_exp, axis=-1), axis=-1)
        self.p = x_exp / x_exp_sum
        y_data = - np.sum(np.log(self.p) * t.data)
        y = Tensor([1])
        y.data = y_data
        self.y = y
        y.om = self
        return y

    def backward(self):
        self.x.grad = 2 * (self.p - self.t.data)
        self.x.backward()

    def __call__(self, x: Tensor, t: Tensor):
        return self.forward(x, t)


def softmax(x: Tensor):
    x_exp = np.exp(x.data)
    x_exp_sum = np.expand_dims(np.sum(x_exp, axis=-1), axis=-1)
    p = x_exp / x_exp_sum
    return p


class Module(object):
    def __init__(self):
        self.parameters = []
        self.expected_module_classes = [Linear, Conv2D]
        self.modules = []

    def initialize(self):
        self.parameters.clear()
        self.modules = inspect.getmembers(self)
        for m in self.modules:
            for expected_module_class in self.expected_module_classes:
                attribute = getattr(self, m[0])
                if isinstance(attribute, expected_module_class):
                    for parameter in attribute.parameters:
                        self.parameters.append(parameter)

    def forward(self, x: Tensor):
        return x

    def __call__(self, x: Tensor):
        return self.forward(x)
