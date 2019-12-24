from typing import Callable

import torch
import torch.nn as nn
from torchsso.utils import *

from opt_einsum import contract


def forward_postprocess(module, input, output):
    data_input = input[0].clone().detach()

    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        bnorm = module
        if not bnorm.affine:
            return
        f = bnorm.num_features
        if isinstance(module, nn.BatchNorm1d):
            shape = (1, f)
        elif isinstance(module, nn.BatchNorm2d):
            shape = (1, f, 1, 1)
        else:
            shape = (1, f, 1, 1, 1)
        # restore normalized input
        data_input_norm = (output - bnorm.bias.view(shape)).div(bnorm.weight.view(shape))
        data_input = data_input_norm

    if isinstance(module, nn.LayerNorm):
        layernorm = module
        if not layernorm.elementwise_affine:
            return
        # restore normalized input
        data_input_norm = (output - layernorm.bias).div(layernorm.weight)
        data_input = data_input_norm

    def backward_hook(grad_output):
        grad_output = grad_output.clone().detach()
        assert data_input.size(0) == grad_output.size(0)

        args = [module, data_input, grad_output]
        if isinstance(module, nn.Linear):
            grad_linear(*args)
        elif isinstance(module, nn.Conv1d):
            grad_conv1d(*args)
        elif isinstance(module, nn.Conv2d):
            grad_conv2d(*args)
        elif isinstance(module, nn.Conv3d):
            grad_conv3d(*args)
        elif isinstance(module, nn.ConvTranspose1d):
            grad_conv_transpose1d(*args)
        elif isinstance(module, nn.ConvTranspose2d):
            grad_conv_transpose2d(*args)
        elif isinstance(module, nn.ConvTranspose3d):
            grad_conv_transpose3d(*args)
        elif isinstance(module, nn.BatchNorm1d):
            grad_batchnorm1d(*args)
        elif isinstance(module, nn.BatchNorm2d):
            grad_batchnorm2d(*args)
        elif isinstance(module, nn.BatchNorm3d):
            grad_batchnorm3d(*args)
        elif isinstance(module, nn.LayerNorm):
            grad_layernorm(*args)
        elif isinstance(module, nn.Embedding):
            grad_embedding(*args)
        else:
            raise ValueError(f'Unsupported module class: {module.__class__}.')

    if output.requires_grad:
        output.register_hook(backward_hook)


def grad_linear(module: nn.Module, data_input: torch.Tensor, grad_output: torch.Tensor):

    assert isinstance(module, nn.Linear)
    linear = module
    assert data_input.ndimension() == 2  # n x f_in
    assert grad_output.ndimension() == 2  # n x f_out

    if linear.weight.requires_grad:
        grads = contract('bi,bj->bij', grad_output, data_input)  # n x f_out x f_in
        setattr(linear.weight, 'grads', grads)  # n x f_out x f_in

    if linear.bias is not None and linear.bias.requires_grad:
        setattr(linear.bias, 'grads', grad_output)  # n x f_out


def _grad_conv(module: nn.Module, data_input: torch.Tensor, grad_output: torch.Tensor, im2col_func: Callable):

    if module.weight.requires_grad:
        # kernel_size = (k_1)(k_2)...(k_d)
        # n x (c_in)(kernel_size) x output_size
        input2d = im2col_func(data_input, module)

        n, c_out = grad_output.size()[0:2]
        grad_output2d = grad_output.view(n, c_out, -1)  # n x c_out x output_size

        grads_2d = contract('bik,bjk->bij', grad_output2d, input2d)  # n x c_out x (c_in)(kernel_size)
        setattr(module.weight, 'grads', grads_2d.view(n, *module.weight.size()))  # n x c_out x c_in x k_1 x ... x k_d

    if module.bias is not None and module.bias.requires_grad:
        ndim = grad_output.ndimension()
        setattr(module.bias, 'grads', grad_output.sum(dim=tuple(range(ndim))[2:]))  # n x c_out


def grad_conv1d(module: nn.Module, data_input: torch.Tensor, grad_output: torch.Tensor):

    assert isinstance(module, nn.Conv1d)
    assert data_input.ndimension() == 3  # n x c_in x l_in
    assert grad_output.ndimension() == 3  # n x c_out x l_out

    _grad_conv(module, data_input, grad_output, im2col_1d)


def grad_conv2d(module: nn.Module, data_input: torch.Tensor, grad_output: torch.Tensor):

    assert isinstance(module, nn.Conv2d)
    assert data_input.ndimension() == 4  # n x c_in x h_in x w_in
    assert grad_output.ndimension() == 4  # n x c_out x h_out x w_out

    _grad_conv(module, data_input, grad_output, im2col_2d)


def grad_conv3d(module: nn.Module, data_input: torch.Tensor, grad_output: torch.Tensor):

    assert isinstance(module, nn.Conv3d)
    assert data_input.ndimension() == 5  # n x c_in x t_in x h_in x w_in
    assert grad_output.ndimension() == 5  # n x c_out x t_out x h_out x w_out

    _grad_conv(module, data_input, grad_output, im2col_3d)


def _grad_conv_transpose(module: nn.Module, data_input: torch.Tensor, grad_output: torch.Tensor,
                         im2col_func: Callable):

    if module.weight.requires_grad:
        n, c_in = data_input.size()[0:2]

        # n x c_in x input_size
        input2d = data_input.view(n, c_in, -1)

        # kernel_size = (k_1)(k_2)...(k_d)
        # n x (c_out)(kernel_size) x input_size
        grad_output2d = im2col_func(grad_output, module)

        # n x c_in x (c_out)(kernel_size)
        grads_2d = contract('bik,bjk->bij', input2d, grad_output2d)
        # n x c_in x c_out x k_h x k_w
        setattr(module.weight, 'grads', grads_2d.view(n, *module.weight.size()))

    if module.bias is not None and module.bias.requires_grad:
        ndim = grad_output.ndimension()
        setattr(module.bias, 'grads', grad_output.sum(dim=tuple(range(ndim))[2:]))  # n x c_out


def grad_conv_transpose1d(module: nn.Module, data_input: torch.Tensor, grad_output: torch.Tensor):

    assert isinstance(module, nn.ConvTranspose1d)
    assert data_input.ndimension() == 3  # n x c_in x l_in
    assert grad_output.ndimension() == 3  # n x c_out x l_out

    _grad_conv_transpose(module, data_input, grad_output, im2col_1d)


def grad_conv_transpose2d(module: nn.Module, data_input: torch.Tensor, grad_output: torch.Tensor):

    assert isinstance(module, nn.ConvTranspose2d)
    assert data_input.ndimension() == 4  # n x c_in x h_in x w_in
    assert grad_output.ndimension() == 4  # n x c_out x h_out x w_out

    _grad_conv_transpose(module, data_input, grad_output, im2col_2d)


def grad_conv_transpose3d(module: nn.Module, data_input: torch.Tensor, grad_output: torch.Tensor):

    assert isinstance(module, nn.ConvTranspose3d)
    assert data_input.ndimension() == 5  # n x c_in x t_in x h_in x w_in
    assert grad_output.ndimension() == 5  # n x c_out x t_out x h_out x w_out

    _grad_conv_transpose(module, data_input, grad_output, im2col_3d)


def grad_batchnorm1d(module: nn.Module, data_input: torch.Tensor, grad_output: torch.Tensor):
    assert isinstance(module, nn.BatchNorm1d)
    batchnorm1d = module
    assert data_input.ndimension() == 2  # n x f
    assert grad_output.ndimension() == 2  # n x f
    assert batchnorm1d.affine

    if batchnorm1d.weight.requires_grad:
        grads = data_input.mul(grad_output)  # n x f
        setattr(batchnorm1d.weight, 'grads', grads)

    if batchnorm1d.bias.requires_grad:
        setattr(batchnorm1d.bias, 'grads', grad_output)  # n x f


def grad_batchnorm2d(module: nn.Module, data_input: torch.Tensor, grad_output: torch.Tensor):
    assert isinstance(module, nn.BatchNorm2d)
    batchnorm2d = module
    assert data_input.ndimension() == 4  # n x c x h x w
    assert grad_output.ndimension() == 4  # n x c x h x w
    assert batchnorm2d.affine

    if batchnorm2d.weight.requires_grad:
        grads = data_input.mul(grad_output).sum(dim=(2, 3))  # n x c
        setattr(batchnorm2d.weight, 'grads', grads)

    if batchnorm2d.bias.requires_grad:
        setattr(batchnorm2d.bias, 'grads', grad_output.sum(dim=(2, 3)))  # n x c


def grad_batchnorm3d(module: nn.Module, data_input: torch.Tensor, grad_output: torch.Tensor):
    assert isinstance(module, nn.BatchNorm3d)
    batchnorm3d = module
    assert data_input.ndimension() == 5  # n x c x d x h x w
    assert grad_output.ndimension() == 5  # n x c x d x h x w
    assert batchnorm3d.affine

    if batchnorm3d.weight.requires_grad:
        grads = data_input.mul(grad_output).sum(dim=(2, 3, 4))  # n x c
        setattr(batchnorm3d.weight, 'grads', grads)

    if batchnorm3d.bias.requires_grad:
        setattr(batchnorm3d.bias, 'grads', grad_output.sum(dim=(2, 3, 4)))  # n x c


def grad_layernorm(module: nn.Module, data_input: torch.Tensor, grad_output: torch.Tensor):
    assert isinstance(module, nn.LayerNorm)
    layernorm = module
    assert layernorm.elementwise_affine

    if layernorm.weight.requires_grad:
        grads = data_input.mul(grad_output)
        setattr(layernorm.weight, 'grads', grads)

    if layernorm.bias.requires_grad:
        setattr(layernorm.bias, 'grads', grad_output)


def grad_embedding(module: nn.Module, data_input: torch.Tensor, grad_output: torch.Tensor):
    assert isinstance(module, nn.Embedding)
    embedding = module
    assert data_input.ndimension() + 1 == grad_output.ndimension()

    if embedding.weight.requires_grad:
        grads = torch.zeros(data_input.nelement(),
                            embedding.num_embeddings,
                            embedding.embedding_dim,
                            dtype=grad_output.dtype)

        grad_output = grad_output.flatten(0, -2)
        for i, idx in enumerate(data_input.flatten()):
            grads[i][idx] = grad_output[i]

        grads = grads.view(*data_input.size(),
                           embedding.num_embeddings,
                           embedding.embedding_dim)

        setattr(embedding.weight, 'grads', grads)

