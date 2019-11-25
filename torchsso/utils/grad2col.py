import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Tuple


def grad2col(gy: torch.Tensor, module: nn.Module, input_size: Tuple[int]):
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        return grad2col_1d(gy, module, input_size)
    elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        return grad2col_2d(gy, module, input_size)
    elif isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
        return grad2col_3d(gy, module, input_size)
    else:
        raise ValueError(f'Unsupported module: {module}.')


def grad2col_1d(gy: torch.Tensor, conv1d: nn.Module, input_size: Tuple[int]):
    assert len(input_size) == 1  # l_in
    assert gy.ndimension() == 3  # n x c_out x l_out
    assert isinstance(conv1d, (nn.Conv1d, nn.ConvTranspose1d))
    assert conv1d.dilation == (1,), 'dilation > 1 is not supported.'

    l_in, = input_size
    n, c_out, l_out = gy.size()
    stride = conv1d.stride
    dilation = conv1d.dilation
    pad = conv1d.padding
    kernel = conv1d.kernel_size

    w = gy.new_zeros(stride)
    w[0] = 1
    w_ex = w.expand(c_out, 1, *stride)
    gy_ex = F.conv_transpose1d(gy, w_ex, stride=stride, groups=c_out)
    if stride[0] > 1:
        gy_ex = gy_ex[:, :, :-(stride[0]-1)]

    # get padding sizes for gy_ex
    pad_l = kernel[0] - pad[0] - 1
    assert pad_l >= 0
    val_l = l_in + 2 * pad[0] - dilation[0] * (kernel[0] - 1) - 1
    extra_pad_l = val_l % stride[0]

    # padding for gy_ex
    pad_left = pad_l
    pad_right = pad_l + extra_pad_l
    gy_ex = F.pad(gy_ex, [pad_left, pad_right])

    # n x c_out x l_in x k_l
    gy_slices = gy_ex.unfold(2, kernel[0], 1)

    # flip arrays in kernel
    array = np.flip(gy_slices.numpy(), (-1)).copy()
    Mgy = torch.from_numpy(array)

    # n x (c_out)(k_l) x l_in
    Mgy = Mgy.transpose(2, 3).flatten(start_dim=1, end_dim=2)

    return Mgy


def grad2col_2d(gy: torch.Tensor, conv2d: nn.Module, input_size: Tuple[int]):
    assert len(input_size) == 2  # h_in x w_in
    assert gy.ndimension() == 4  # n x c_out x h_out x w_out
    assert isinstance(conv2d, (nn.Conv2d, nn.ConvTranspose2d))
    assert conv2d.dilation == (1, 1), 'dilation > 1 is not supported.'

    h_in, w_in = input_size
    n, c_out, h_out, w_out = gy.size()
    stride = conv2d.stride
    dilation = conv2d.dilation
    pad = conv2d.padding
    kernel = conv2d.kernel_size

    w = gy.new_zeros(stride)
    w[0, 0] = 1
    w_ex = w.expand(c_out, 1, *stride)
    gy_ex = F.conv_transpose2d(gy, w_ex, stride=stride, groups=c_out)
    if stride[0] > 1:
        gy_ex = gy_ex[:, :, :-(stride[0]-1), :]
    if stride[1] > 1:
        gy_ex = gy_ex[:, :, :, :-(stride[1]-1)]

    # get padding sizes for gy_ex
    pad_h = kernel[0] - pad[0] - 1
    pad_w = kernel[1] - pad[1] - 1
    assert pad_h >= 0 and pad_w >= 0
    val_h = h_in + 2 * pad[0] - dilation[0] * (kernel[0] - 1) - 1
    val_w = w_in + 2 * pad[1] - dilation[1] * (kernel[1] - 1) - 1
    extra_pad_h = val_h % stride[0]
    extra_pad_w = val_w % stride[1]

    # padding for gy_ex
    pad_top, pad_left = pad_h, pad_w
    pad_bottom, pad_right = pad_h + extra_pad_h,  pad_w + extra_pad_w
    gy_ex = F.pad(gy_ex, [pad_top, pad_bottom, pad_left, pad_right])

    # n x c_out x h_in x w_in x k_h x k_w
    gy_slices = gy_ex.unfold(2, kernel[0], 1).unfold(3, kernel[1], 1)
    # n x c_out x (h_in)(w_in) x k_h x k_w
    Mgy = gy_slices.flatten(start_dim=2, end_dim=3)

    # flip arrays in kernel
    array = np.flip(Mgy.numpy(), (-1, -2)).copy()
    Mgy = torch.from_numpy(array)

    # n x c_out x (h_in)(w_in) x (k_h)(k_w)
    Mgy = Mgy.flatten(start_dim=3)
    # n x (c_out)(k_h)(k_w) x (h_in)(w_in)
    Mgy = Mgy.transpose(2, 3).flatten(start_dim=1, end_dim=2)

    return Mgy


def grad2col_3d(gy: torch.Tensor, conv3d: nn.Module, input_size: Tuple[int]):
    assert len(input_size) == 3  # t_in x h_in x w_in
    assert gy.ndimension() == 5  # n x c_out x t_out x h_out x w_out
    assert isinstance(conv3d, (nn.Conv3d, nn.ConvTranspose3d))
    assert conv3d.dilation == (1, 1, 1), 'dilation > 1 is not supported.'

    t_in, h_in, w_in = input_size
    n, c_out, t_out, h_out, w_out = gy.size()
    stride = conv3d.stride
    dilation = conv3d.dilation
    pad = conv3d.padding
    kernel = conv3d.kernel_size

    w = gy.new_zeros(stride)
    w[0, 0, 0] = 1
    w_ex = w.expand(c_out, 1, *stride)
    gy_ex = F.conv_transpose3d(gy, w_ex, stride=stride, groups=c_out)
    if stride[0] > 1:
        gy_ex = gy_ex[:, :, :-(stride[0]-1), :, :]
    if stride[1] > 1:
        gy_ex = gy_ex[:, :, :, :-(stride[1]-1), :]
    if stride[2] > 1:
        gy_ex = gy_ex[:, :, :, :, :-(stride[2]-1)]

    # get padding sizes for gy_ex
    pad_t = kernel[0] - pad[0] - 1
    pad_h = kernel[1] - pad[1] - 1
    pad_w = kernel[2] - pad[2] - 1
    assert pad_t >= 0 and pad_h >= 0 and pad_w >= 0
    val_t = t_in + 2 * pad[0] - dilation[0] * (kernel[0] - 1) - 1
    val_h = h_in + 2 * pad[1] - dilation[1] * (kernel[1] - 1) - 1
    val_w = w_in + 2 * pad[2] - dilation[2] * (kernel[2] - 1) - 1
    extra_pad_t = val_t % stride[0]
    extra_pad_h = val_h % stride[1]
    extra_pad_w = val_w % stride[2]

    # padding for gy_ex
    pad_front, pad_top, pad_left = pad_t, pad_h, pad_w
    pad_back, pad_bottom, pad_right = pad_t + extra_pad_t, pad_h + extra_pad_h,  pad_w + extra_pad_w
    gy_ex = F.pad(gy_ex, [pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right])

    # n x c_out x t_in x h_in x w_in x k_t x k_h x k_w
    gy_slices = gy_ex.unfold(2, kernel[0], 1).unfold(3, kernel[1], 1).unfold(4, kernel[2], 1)
    # n x c_out x (t_in)(h_in)(w_in) x k_t x k_h x k_w
    Mgy = gy_slices.flatten(start_dim=2, end_dim=4)

    # flip arrays in kernel
    array = np.flip(Mgy.numpy(), (-1, -2, -3)).copy()
    Mgy = torch.from_numpy(array)

    # n x c_out x (t_in)(h_in)(w_in) x (k_t)(k_h)(k_w)
    Mgy = Mgy.flatten(start_dim=3)
    # n x (c_out)(k_t)(k_h)(k_w) x (t_in)(h_in)(w_in)
    Mgy = Mgy.transpose(2, 3).flatten(start_dim=1, end_dim=2)

    return Mgy

