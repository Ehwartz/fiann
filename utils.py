# -*- coding: utf-8 -*-
# @Author  : Ehwartz
# @Github  : https://github.com/Ehwartz
# @Time    : 05/21/2024
# @Software: PyCharm
# @File    : utils.py
import numpy as np


def fold(unfolded, output_size, kernel_size, stride, padding):
    n, c, h, w = output_size
    out_h = (h + 2 * padding - kernel_size) // stride + 1
    out_w = (w + 2 * padding - kernel_size) // stride + 1

    output_array = np.zeros((n, c, h + 2 * padding, w + 2 * padding))
    count_matrix = np.zeros_like(output_array)

    unfolded = unfolded.reshape(n, out_h, out_w, c, kernel_size, kernel_size)
    unfolded = unfolded.transpose(0, 3, 4, 5, 1, 2)

    h_indices = np.arange(out_h) * stride
    w_indices = np.arange(out_w) * stride

    for i in range(kernel_size):
        for j in range(kernel_size):
            output_array[:, :, h_indices[:, None] + i, w_indices + j] += unfolded[:, :, i, j, :, :]
            count_matrix[:, :, h_indices[:, None] + i, w_indices + j] += 1

    output_array /= count_matrix
    if padding > 0:
        output_array = output_array[:, :, padding:-padding, padding:-padding]

    return output_array


def fold_grad(unfolded, output_size, kernel_size, stride, padding):
    n, c, h, w = output_size
    out_h = (h + 2 * padding - kernel_size) // stride + 1
    out_w = (w + 2 * padding - kernel_size) // stride + 1

    output_array = np.zeros((n, c, h + 2 * padding, w + 2 * padding))

    unfolded = unfolded.reshape(n, out_h, out_w, c, kernel_size, kernel_size)
    unfolded = unfolded.transpose(0, 3, 4, 5, 1, 2)

    h_indices = np.arange(out_h) * stride
    w_indices = np.arange(out_w) * stride

    for i in range(kernel_size):
        for j in range(kernel_size):
            output_array[:, :, h_indices[:, None] + i, w_indices + j] += unfolded[:, :, i, j, :, :]

    if padding > 0:
        output_array = output_array[:, :, padding:-padding, padding:-padding]

    return output_array


def unfold(input_array, kernel_size, stride, padding):
    input_padded = np.pad(input_array, pad_width=((0, 0), (0, 0),
                                                  (padding, padding),
                                                  (padding, padding)), mode='constant')

    n, c, h, w = input_padded.shape
    out_h = (h - kernel_size) // stride + 1
    out_w = (w - kernel_size) // stride + 1

    unfolded = np.lib.stride_tricks.as_strided(
        input_padded,
        shape=(n, c, out_h, out_w, kernel_size, kernel_size),
        strides=(
            input_padded.strides[0],
            input_padded.strides[1],
            input_padded.strides[2] * stride,
            input_padded.strides[3] * stride,
            input_padded.strides[2],
            input_padded.strides[3]
        )
    )

    unfolded = unfolded.reshape([n, c, -1, kernel_size * kernel_size]
                                ).transpose(0, 2, 1, 3).reshape(n, out_h * out_w, c * kernel_size * kernel_size)
    return unfolded


def valid(model, dataloader):
    tot = 0
    cor = 0
    for i, (x, t) in enumerate(dataloader):
        tot += x.shape[0]
        y = model(x)
        cor += np.sum((np.argmax(y.data, axis=-1) == np.argmax(t.data, axis=-1)) * 1)

    return cor / tot, cor, tot
