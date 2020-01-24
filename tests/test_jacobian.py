import torch
import torch.nn as nn
import torch.nn.functional as F

from opt_einsum import contract

import time


class Net(nn.Module):

    def __init__(self, n_inputs=3, n_outputs=10):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, 100, bias=False)
        self.fc2 = nn.Linear(100, n_outputs, bias=False)

    def forward(self, x):
        s1 = self.fc1(x)
        a1 = F.relu(s1)
        s2 = self.fc2(a1)

        return s2


class ConvNet(nn.Module):

    def __init__(self, n_outputs=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, bias=False)
        self.conv2 = nn.Conv2d(128, 256, 3, bias=False)
        self.fc3 = nn.Linear(200704, n_outputs, bias=False)

    def forward(self, x):
        s1 = self.conv1(x)
        a1 = F.relu(s1)
        s2 = self.conv2(a1)
        a2 = F.relu(s2)
        a2 = a2.flatten(start_dim=1)
        s3 = self.fc3(a2)

        return s3


def register_hooks(model: nn.Module):

    def forward_postprocess(module, input, output):
        data_input = input[0]

        setattr(module, 'data_input', data_input)
        setattr(module, 'data_output', output)

    for module in model.children():
        module.register_forward_hook(forward_postprocess)


def manual_jacobian(x, model):
    # x -> fc1 -> s1
    # s1 -> relu -> a1
    # a1 -> fc2 -> s2 (output)

    register_hooks(model)
    output = model(x)

    n = output.size(0)
    fc1 = model.fc1
    fc2 = model.fc2

    # Jacobian s2 (output) -> a1
    Ja1 = torch.stack([fc2.weight] * n)

    # ReLU derivative a1 -> s1
    d = (fc2.data_input > 0).type(fc2.data_input.dtype)

    # Jacobian s2 (output) -> s1
    Js1 = contract('bij,bj->bij', Ja1, d)

    # Jacobian s2 (output) -> x
    Jx = contract('bij,jk->bik', Js1, fc1.weight)

    return Jx


def manual_jacobian_conv(x, model):
    # x -> conv1 -> s1
    # s1 -> relu -> a1
    # a1 -> conv2 -> s2
    # s2 -> relu -> a2
    # a2 -> fc3 -> s3 (output)

    bs = x.shape[0]

    register_hooks(model)
    output = model(x)
    n_outputs = output.shape[-1]

    conv1 = model.conv1
    conv2 = model.conv2
    fc3 = model.fc3

    # Jacobian s3 (output) -> a2
    w3 = fc3.weight
    Ja2 = torch.stack([w3] * bs)

    # ReLU derivative a2 -> s2
    a2 = fc3.data_input
    d = (a2 > 0).type(a2.dtype)

    # Jacobian s3 (output) -> s2
    Js2 = contract('bij,bj->bij', Ja2, d)

    # Jacobian s3 (output) -> a1
    a1 = conv2.data_input
    s2 = conv2.data_output
    Js2 = Js2.reshape(bs * n_outputs, *s2.shape[1:])
    Ja1 = F.grad.conv2d_input(a1.shape, conv2.weight, Js2)

    # ReLU derivative a1 -> s1
    d = (a1 > 0).type(a1.dtype)

    # Jacobian s3 (output) -> s1
    Ja1 = Ja1.reshape(bs, n_outputs, *a1.shape[1:])
    Js1 = contract('bichw,bchw->bichw', Ja1, d)

    # Jacobian s3 (output) -> x
    s1 = conv1.data_output
    Js1 = Js1.reshape(bs * n_outputs, *s1.shape[1:])
    Jx = F.grad.conv2d_input(x.shape, conv1.weight, Js1)

    return Jx


def reverse_mode_jacobian_with_repeat(x, model, bs=10, n_outputs=10):
    repeat_arg = (n_outputs,) + (1,) * len(x.size())
    xr = x.repeat(*repeat_arg)
    xr = xr.transpose(0, 1)
    xr.requires_grad_(True)
    output = model(xr)
    assert n_outputs == output.shape[-1]
    I = torch.eye(n_outputs, device=xr.device)
    I = I.repeat(bs, 1, 1)

    Jx = torch.autograd.grad(output, xr, grad_outputs=I)[0]

    return Jx


def reverse_mode_jacobian_with_repeat_conv(x, model, n_outputs=10):
    bs = x.shape[0]
    repeat_arg = (n_outputs,) + (1,) * len(x.size())
    xr = x.repeat(*repeat_arg)
    xr = xr.transpose(0, 1)
    xr = xr.reshape(bs * n_outputs, *x.shape[1:])
    xr.requires_grad_(True)
    output = model(xr)
    output = output.reshape(bs, n_outputs, -1)
    assert n_outputs == output.shape[-1]
    I = torch.eye(n_outputs, device=xr.device)
    I = I.repeat(bs, 1, 1)

    Jx = torch.autograd.grad(output, xr, grad_outputs=I)[0]

    return Jx


def test_jacobian():
    bs = 128
    n_inputs = 1000
    n_outputs = 1000
    loop = 1
    x = torch.randn(bs, n_inputs)
    model = Net(n_inputs, n_outputs)

    print(f'bs: {bs}')
    print(f'n_inputs: {n_inputs}')
    print(f'n_outputs: {n_outputs}')
    print(f'loop: {loop}')
    print('-------------')

    start = time.time()
    for i in range(loop):
        Jx_rev = reverse_mode_jacobian_with_repeat(x, model, bs, n_outputs)
    elapsed = time.time() - start
    print(f'reverse mode: {elapsed:.3f}s')

    start = time.time()
    for i in range(loop):
        Jx_man = manual_jacobian(x, model)
    elapsed = time.time() - start
    print(f'manual mode: {elapsed:.3f}s')

    print(f'(Jx_rev - Jx_man).max(): {(Jx_rev - Jx_man).max()}')


def test_jacobian_conv():
    bs = 128
    n_outputs = 100
    model = ConvNet(n_outputs)
    x = torch.randn(bs, 3, 32, 32)
    loop = 1

    print(f'bs: {bs}')
    print(f'input shape: {x.shape[1:]}')
    print(f'n_outputs: {n_outputs}')
    print(f'loop: {loop}')
    print('-------------')

    start = time.time()
    for i in range(loop):
        Jx_rev = reverse_mode_jacobian_with_repeat_conv(x, model, n_outputs)
    elapsed = time.time() - start
    print(f'reverse mode: {elapsed:.3f}s')

    start = time.time()
    for i in range(loop):
        Jx_man = manual_jacobian_conv(x, model)
    elapsed = time.time() - start
    print(f'manual mode: {elapsed:.3f}s')

    print(f'(Jx_rev - Jx_man).max(): {(Jx_rev - Jx_man).max()}')


if __name__ == '__main__':
    test_jacobian()
    print('*****************')
    test_jacobian_conv()
