from typing import List
from contextlib import contextmanager
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsso.autograd import utils
from torchsso.autograd import operations


@contextmanager
def extend(model, op_names):

    handles = []

    def forward_hook(module, in_data, out_data):
        in_data = in_data[0].clone().detach()
        _call_operations_in_forward(module, in_data)

        def backward_hook(out_grads):
            out_grads = out_grads.clone().detach()
            _call_operations_in_backward(module, in_data, out_grads)

        if out_data.requires_grad:
            handles.append(out_data.register_hook(backward_hook))

    # register hooks and operations in modules
    for module in model.modules():
        if len(list(module.children())) > 0:
            continue
        params = list(module.parameters())
        params = [p for p in params if p.requires_grad]
        if len(params) == 0:
            continue
        handles.append(module.register_forward_hook(forward_hook))
        _register_operations(module, op_names)

    yield

    # remove hooks and operations from modules
    for handle in handles:
        handle.remove()
    for module in model.modules:
        _remove_operations(module)


def _register_operations(module: nn.Module, op_names: List):
    module_class_name = module.__class__.__name__
    op_class = getattr(operations, module_class_name, None)
    if op_class is not None:
        setattr(module, 'operation', op_class(module, op_names))
    else:
        warnings.warn(f'Failed to lookup operations for Module {module_class_name}.')


def _call_operations_in_forward(module, in_data):
    operation = getattr(module, 'operation', None)
    if operations is not None:
        operation.forward_post_process(in_data)


def _call_operations_in_backward(module, in_data, out_grads):
    operation = getattr(module, 'operation', None)
    if operation is not None:
        operation.backward_pre_process(in_data, out_grads)


def _remove_operations(module):
    if hasattr(module, 'operation'):
        delattr(module, 'operation')


def fisher_cross_entropy(model, inputs, fisher_types,
                         targets=None, requires_param_grad=True):
    logits = model(inputs)
    n_examples, n_classes = logits.shape

    # empirical
    if 'Emp' in fisher_types:
        assert targets is not None
        loss = F.cross_entropy(logits, targets)
        if requires_param_grad:
            loss.backward(retain_graph=True)
        else:
            with utils.disable_param_grad(model):
                loss.backward(retain_graph=True)

    probs = F.softmax(logits, dim=1)

    if 'MC' in fisher_types:
        dist = torch.distributions.Categorical(probs)
        _targets = dist.sample((n_samples,))
        # for-loop
        with utils.disable_param_grad(model):
            loss.backward(retain_graph=True)
            # scale & accumulate results to somewhere

    if 'Exact' in fisher_types:
        # for-loop
        with utils.disable_param_grad(model):
            loss.backward(retain_graph=True)
            # scale & accumulate results to somewhere

