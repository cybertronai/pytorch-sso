from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextmanager
def save_sample_grads(model: nn.Module):

    handles = []
    for module in model.children():
        params = list(module.parameters())
        params = [p for p in params if p.requires_grad]
        if len(params) == 0:
            continue

        handles.append(module.register_forward_hook(_forward_postprocess))
        handles.append(module.register_backward_hook(_backward_postprocess))

    yield
    for handle in handles:
        handle.remove()


def _forward_postprocess(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
    data_input = input[0].clone().detach()

    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        bnorm = module
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

    setattr(module, 'data_input', data_input)


def _backward_postprocess(module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor):
    grad_output = grad_output[0].clone().detach()
    data_input = getattr(module, 'data_input', None)
    assert data_input is not None, 'backward is called before forward.'
    assert data_input.size(0) == grad_output.size(0)

    args = [module, data_input, grad_output]
    if isinstance(module, nn.Linear):
        grad_linear(*args)
    elif isinstance(module, nn.Conv2d):
        grad_conv2d(*args)
    elif isinstance(module, nn.BatchNorm1d):
        grad_batchnorm1d(*args)
    elif isinstance(module, nn.BatchNorm2d):
        grad_batchnorm2d(*args)
    else:
        raise ValueError(f'Unsupported module class: {module.__class__}.')


def grad_linear(module: nn.Module, data_input: torch.Tensor, grad_output: torch.Tensor):

    assert isinstance(module, nn.Linear)
    linear = module
    assert data_input.ndimension() == 2  # n x f_in
    assert grad_output.ndimension() == 2  # n x f_out

    if linear.weight.requires_grad:
        grads = torch.einsum('bi,bj->bij', grad_output, data_input)  # n x f_out x f_in
        setattr(linear.weight, 'grads', grads)  # n x f_out x f_in

    if hasattr(linear, 'bias') and linear.bias.requires_grad:
        setattr(linear.bias, 'grads', grad_output)  # n x f_out


def grad_conv2d(module: nn.Module, data_input: torch.Tensor, grad_output: torch.Tensor):

    assert isinstance(module, nn.Conv2d)
    conv2d = module
    assert data_input.ndimension() == 4  # n x c_in x h_in x w_in
    assert grad_output.ndimension() == 4  # n x c_out x h_out x w_out

    if conv2d.weight.requires_grad:
        # n x (c_in)(k_h)(k_w) x (h_out)(w_out)
        input2d = F.unfold(data_input,
                           kernel_size=conv2d.kernel_size, stride=conv2d.stride,
                           padding=conv2d.padding, dilation=conv2d.dilation)

        # n x c_out x h_out x w_out
        n, c_out, h, w = grad_output.size()
        # n x c_out x (h_out)(w_out)
        grad_output2d = grad_output.view(n, c_out, -1)

        c_out, c_in, k_h, k_w = conv2d.weight.size()

        grads_2d = torch.einsum('bik,bjk->bij', grad_output2d, input2d)  # n x c_out x (c_in)(k_h)(k_w)
        setattr(conv2d.weight, 'grads', grads_2d.view(n, c_out, c_in, k_h, k_w))  # n x c_out x c_in x k_h x k_w

    if hasattr(conv2d, 'bias') and conv2d.bias.requires_grad:
        setattr(conv2d.bias, 'grads', grad_output.sum(dim=(2, 3)))  # n x c_out


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

