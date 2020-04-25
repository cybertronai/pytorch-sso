from typing import List
from contextlib import contextmanager
import warnings

import torch.nn as nn
from torchsso.autograd.utils import record_original_requires_grad
from torchsso.autograd.operations import Operation, OP_KRON, OP_DIAG, OP_BATCH_GRADS  # NOQA
from torchsso.autograd.fisher import fisher_for_cross_entropy, FISHER_EXACT, FISHER_MC, FISHER_EMP  # NOQA


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

    for module in model.modules():
        if len(list(module.children())) > 0:
            continue
        requires_grad = False
        for param in module.parameters():
            requires_grad = requires_grad or param.requires_grad
            record_original_requires_grad(param)
        if not requires_grad:
            continue
        # register hooks and operations in modules
        handles.append(module.register_forward_hook(forward_hook))
        _register_operations(module, op_names)

    yield

    # remove hooks and operations from modules
    for handle in handles:
        handle.remove()
    for module in model.modules():
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

