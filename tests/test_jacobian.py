import sys
import torch
import torch.nn as nn

import flax
import jax
import jax.numpy as jnp

from timeit import timeit


class Net(nn.Module):

    def __init__(self, n_inputs, n_outputs, hidden_ndim, n_layers):
        super().__init__()
        assert n_layers > 2
        layers = [nn.Linear(n_inputs, hidden_ndim, bias=False), nn.ReLU()]
        for i in range(n_layers - 2):
            layers.append(nn.Linear(hidden_ndim, hidden_ndim, bias=False))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_ndim, n_outputs, bias=False))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FlaxNet(flax.nn.Module):

    def apply(self, x, n_outputs, hidden_ndim, n_layers):
        assert n_layers > 2
        x = flax.nn.Dense(x, features=hidden_ndim, bias=False)
        x = flax.nn.relu(x)
        for i in range(n_layers - 2):
            x = flax.nn.Dense(x, features=hidden_ndim, bias=False)
            x = flax.nn.relu(x)
        x = flax.nn.Dense(x, features=n_outputs, bias=False)

        return x


def register_hooks(model: nn.Module):

    targets = (nn.Linear, nn.ReLU)

    def forward_postprocess(module, input, output):
        data_input = input[0]

        setattr(module, 'data_input', data_input)
        setattr(module, 'data_output', output)

    for module in model.modules():
        if isinstance(module, targets):
            module.register_forward_hook(forward_postprocess)


def jax_jacobian(rng, input_shape, model, Jx_fn):
    x = jax.random.normal(rng, input_shape)
    y = model(x)
    J = Jx_fn(x)

    return J


def manual_jacobian(bs, n_inputs, model, device, mode='rev'):
    x = torch.randn(bs, n_inputs, device=device)
    y = model(x)
    modules = []
    for module in model.modules():
        if len(list(module.children())) > 0:
            continue
        modules.append(module)

    if mode == 'rev':
        J = manual_jacobian_rev(modules)
    else:
        J = manual_jacobian_fwd(modules)

    return J


def manual_jacobian_rev(modules):
    J = None

    for module in modules[::-1]:
        if J is None:
            J = module.weight
        elif isinstance(module, nn.ReLU):
            a = module.data_output
            d = (a > 0).type(a.dtype)
            if J.ndimension() == 2:
                J = torch.einsum('ab,nb->nab', J, d)
            else:
                J = torch.einsum('nab,nb->nab', J, d)
        elif isinstance(module, nn.Linear):
            J = torch.einsum('nab,bc->nac', J, module.weight)

    return J


def manual_jacobian_fwd(modules):
    J = None

    for module in modules:
        if J is None:
            J = module.weight
        elif isinstance(module, nn.ReLU):
            a = module.data_output
            d = (a > 0).type(a.dtype)
            if J.ndimension() == 2:
                J = torch.einsum('na,ab->nab', d, J)
            else:
                J = torch.einsum('na,nab->nab', d, J)
        elif isinstance(module, nn.Linear):
            J = torch.einsum('ab,nbc->nac', module.weight, J)

    return J


def reverse_mode_jacobian_with_repeat(bs, n_inputs, n_outputs, model, device):
    x = torch.randn(bs, n_inputs, device=device)

    repeat_arg = (n_outputs,) + (1,) * len(x.size())
    xr = x.repeat(*repeat_arg)
    xr = xr.transpose(0, 1)
    xr.requires_grad_(True)
    y = model(xr)
    I = torch.eye(n_outputs, device=xr.device)
    I = I.repeat(bs, 1, 1)

    Jx = torch.autograd.grad(y, xr, grad_outputs=I, retain_graph=True)[0]

    return Jx


def main(mode):
    bs = 32
    n_inputs = 1000
    hidden_ndim = 1000
    n_outputs = 1000
    n_layers = 10
    loop = 100
    print('-------------')
    print(f'mode: {mode}')
    print(f'bs: {bs}')
    print(f'n_inputs: {n_inputs}')
    print(f'hidden_ndim: {hidden_ndim}')
    print(f'n_outputs: {n_outputs}')
    print(f'n_layers: {n_layers}')
    print('-------------')
    print(f'loop: {loop}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if mode == 'torch.auto':
        # ------------------
        # PyTorch auto-diff
        print(f'device: {device}')

        model = Net(n_inputs, n_outputs, hidden_ndim, n_layers)
        model = model.to(device)
        args = [bs, n_inputs, n_outputs, model, device]
        reverse_mode_jacobian_with_repeat(*args)
        elapsed = timeit(lambda: reverse_mode_jacobian_with_repeat(*args), number=loop)
        print(f'torch auto rev: {elapsed:.3f}s')
    elif mode == 'torch.man':
        # ------------------
        # PyTorch manual-diff
        print(f'device: {device}')

        model = Net(n_inputs, n_outputs, hidden_ndim, n_layers)
        model = model.to(device)
        register_hooks(model)
        args = [bs, n_inputs, model, device]
        manual_jacobian(*args)
        elapsed = timeit(lambda: manual_jacobian(*args), number=loop)
        print(f'torch manual rev: {elapsed:.3f}s')
    else:
        # ------------------
        # JAX
        print(f'device: {jax.devices()}')

        rng = jax.random.PRNGKey(0)
        input_shape = (bs, n_inputs)
        model_def = FlaxNet.partial(n_outputs=n_outputs, hidden_ndim=hidden_ndim, n_layers=n_layers)
        _, model = model_def.create_by_shape(rng, [(input_shape, jnp.float32)])

        jac_fun = jax.jit(jax.jacrev(model))
        Jx_fn = jax.jit(jax.vmap(jac_fun, in_axes=(0,)))
        args = [rng, input_shape, model, Jx_fn]
        jax_jacobian(*args)
        elapsed = timeit(lambda: jax_jacobian(*args), number=loop)
        print(f'jit(jax.jacrev): {elapsed:.3f}s')


if __name__ == '__main__':
    mode = sys.argv[1]
    main(mode)
