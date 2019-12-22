from typing import Callable
from contextlib import contextmanager

import torch.nn as nn
from torchsso.autograd import gradient


@contextmanager
def module_hook(model: nn.Module,
                forward_preprocess: Callable=None,
                forward_postprocess: Callable=None):

    handles = []
    for module in model.modules():
        if len(list(module.children())) > 0:
            continue
        params = list(module.parameters())
        params = [p for p in params if p.requires_grad]
        if len(params) == 0:
            continue

        if forward_preprocess is not None:
            handles.append(module.register_forward_pre_hook(forward_preprocess))
        if forward_postprocess is not None:
            handles.append(module.register_forward_hook(forward_postprocess))

    yield
    for handle in handles:
        handle.remove()


def save_batched_grads(model: nn.Module):
    return module_hook(model, forward_postprocess=gradient.forward_postprocess)


