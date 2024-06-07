# -*- coding: utf-8 -*-
# @Author  : Ehwartz
# @Github  : https://github.com/Ehwartz
# @Time    : 06/01/2024
# @Software: PyCharm
# @File    : optim.py
import numpy as np
from tensor import Tensor


class Optimizer(object):
    def __init__(self, params: list[Tensor], lr=0.001, momentum=0,
                 dampening=0, weight_decay=0, nesterov=False):
        self.parameters = params
        self.lr = lr

    def zero_grad(self):
        for param in self.parameters:
            param.grad *= 0

    def step(self):
        for param in self.parameters:
            param.data -= self.lr * param.grad


class SGD(Optimizer):
    def __init__(self, params: list[Tensor], lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(SGD, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.parameters = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.num_params = len(self.parameters)
        self.buffs = []
        self.t = 0
        for i in range(self.num_params):
            self.buffs.append(np.zeros_like(self.parameters[i].data))

    def step(self):
        for i in range(self.num_params):
            param = self.parameters[i]
            g = param.grad
            if self.weight_decay > 0:
                g += self.weight_decay * param.data
            if self.momentum > 0:
                if self.t > 0:
                    self.buffs[i] = self.momentum * self.buffs[i] + (1 - self.momentum) * g
                else:
                    self.buffs[i] = g
            else:
                self.buffs[i] = g
            if self.nesterov:
                g = g + self.momentum * self.buffs[i]
            else:
                g = self.buffs[i]
            self.t += 1
            param.data -= self.lr * g
