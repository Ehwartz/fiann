# -*- coding: utf-8 -*-
# @Author  : Ehwartz
# @Github  : https://github.com/Ehwartz
# @Time    : 06/01/2024
# @Software: PyCharm
# @File    : data.py
import numpy as np
import torchvision
from tensor import Tensor


class Dataset(object):
    def __init__(self):
        self.xs = Tensor([0])
        self.ys = Tensor([0])
        self.n = len(self)

    def __getitem__(self, item):
        return self.xs[item], self.ys[item], item

    def __len__(self):
        return self.xs.shape[0]


class DataLoader(object):
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool):
        self.dataset = dataset
        self.xs = dataset.xs
        self.ys = dataset.ys
        self.batch_size = batch_size
        self.index = np.arange(len(self.dataset))
        self.shuffle = shuffle
        self.queue = self.index.copy()
        if self.shuffle:
            np.random.shuffle(self.queue)

    def __getitem__(self, item):
        if not self.queue.size:
            self._update_index()
            raise StopIteration
        if self.queue.size > self.batch_size:
            indices = self.queue[:self.batch_size]
            self.queue = self.queue[self.batch_size:]
        else:
            indices = self.queue[:]
            self.queue = self.queue[-1:-1]
        return self.xs[indices], self.ys[indices]

    def __len__(self):
        return len(self.dataset)

    def _update_index(self):
        self.queue = self.index.copy()
        if self.shuffle:
            np.random.shuffle(self.queue)


class MNIST(Dataset):
    def __init__(self, root, n=60000):
        super(MNIST, self).__init__()
        mds = torchvision.datasets.MNIST(root=root, download=True)

        data = mds.data.detach().numpy()[:n].reshape([-1, 1, 28, 28])

        self.xs = Tensor(list(data.shape))
        self.ys = Tensor([data.shape[0], 10], 'zeros')
        targets = mds.targets.detach().numpy()[:n]
        self.ys.data[np.arange(data.shape[0]), targets] = 1
