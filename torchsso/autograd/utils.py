from contextlib import contextmanager
import numpy as np
import torch

REQUIRES_GRAD_ATTR = '_original_requires_grad'


def original_requires_grad(module, param_name):
    param = getattr(module, param_name, None)
    return param is not None and getattr(param, REQUIRES_GRAD_ATTR)


def record_original_requires_grad(param):
    setattr(param, REQUIRES_GRAD_ATTR, param.requires_grad)


def restore_original_requires_grad(param):
    param.requires_grad = getattr(param, REQUIRES_GRAD_ATTR, param.requires_grad)


@contextmanager
def disable_param_grad(model):

    for param in model.parameters():
        record_original_requires_grad(param)
        param.requires_grad = False

    yield
    for param in model.parameters():
        restore_original_requires_grad(param)


