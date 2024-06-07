# -*- coding: utf-8 -*-
# @Author  : Ehwartz
# @Github  : https://github.com/Ehwartz
# @Time    : 05/21/2024
# @Software: PyCharm
# @File    : tensor.py

import numpy as np


class Tensor(object):
    def __init__(self, shape, init='random'):
        if init == 'random':
            self.data = 2 * np.random.random(size=shape) - 1
        elif init == 'ones':
            self.data = np.ones(shape=shape)
        elif init == 'zeros':
            self.data = np.zeros(shape=shape)
        else:
            self.data = 2 * np.random.random(size=shape) - 1
        self.grad = np.zeros(shape=shape)
        self.shape = self._shape()
        self.im = None
        self.om = None

    def _shape(self):
        return list(self.data.shape)

    def __getitem__(self, item):
        data = self.data[item]
        grad = self.grad[item]
        ret = Tensor(list(data.shape))
        ret.data = data
        ret.grad = grad
        return ret

    def reshape(self, shape):
        ret = Tensor(shape)
        ret.data = self.data.copy().reshape(shape)
        ret.grad = self.grad.copy().reshape(shape)
        ret.im = self.im
        ret.om = self.om
        return ret

    def transpose(self, axes):
        ret = Tensor(self.shape)
        ret.data = ret.data.transpose(axes)
        ret.grad = ret.grad.transpose(axes)
        ret.im = self.im
        ret.om = self.om
        ret.shape = ret._shape()
        return ret

    def backward(self):
        if self.om:
            self.om.backward()
