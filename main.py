# -*- coding: utf-8 -*-
# @Author  : Ehwartz
# @Github  : https://github.com/Ehwartz
# @Time    : 06/07/2024
# @Software: PyCharm
# @File    : main.py

import numpy as np

import nn
from tensor import Tensor
import data
import optim
import utils


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.LeakyReLU(0.1)
        self.reshape1 = nn.Reshape([-1, 32, 28, 28], [-1, 32 * 28 * 28])
        self.linear1 = nn.Linear(32 * 28 * 28, 10, False)

        self.initialize()

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.reshape1(x)
        x = self.linear1(x)
        return x


if __name__ == '__main__':
    model = Net()
    dataset = data.MNIST('./datasets/mnist', 200)
    dataloader = data.DataLoader(dataset, 64, True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters, lr=1e-3, momentum=0.5, nesterov=False)
    for iep in range(7):
        print('  epoch: ', iep)
        for i, (xs, ts) in enumerate(dataloader):
            optimizer.zero_grad()
            ys = model(xs)
            l = criterion(ys, ts)
            l.backward()
            optimizer.step()
            print("     loss:  ", l.data)
        acc, cor, tot = utils.valid(model, dataloader)
        print(f'acc: {acc:.3}, cor/tot: {cor}/{tot}')
