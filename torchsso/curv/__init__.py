import torch
import torch.nn as nn
import torchsso
from torchsso.utils import TensorAccumulator
from torchsso.curv.curvature import Curvature, DiagCurvature, KronCurvature  # NOQA
from torchsso.curv.cov.linear import CovLinear, DiagCovLinear, KronCovLinear  # NOQA
from torchsso.curv.cov.conv import DiagCovConvNd, DiagCovConvTransposeNd, KronCovConvNd, KronCovConvTransposeNd  # NOQA
from torchsso.curv.cov.batchnorm import DiagCovBatchNormNd  # NOQA
from torchsso.curv.cov.sparse import KronCovEmbedding  # NOQA


def get_curv_class(module: nn.Module, curv_type: str, curv_shape: str):
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        module_name = 'ConvNd'
    elif isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        module_name = 'ConvTransposeNd'
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        module_name = 'BatchNormNd'
    else:
        module_name = module.__class__.__name__

    curv_name = curv_shape + curv_type + module_name
    curv_class = getattr(torchsso.curv, curv_name, None)

    if curv_class is None:
        print(f'[warning] Failed to lookup Curvature class {curv_name} for {module}.')

    return curv_class


class Layer(object):

    def __init__(self, module: nn.Module, name: str, curv: Curvature):

        self.module = module
        self.name = name
        self.curv = curv
        self.acc_curv = TensorAccumulator()


class Observer(object):

    def __init__(self, model: nn.Module, curv_type: str,
                 curv_shapes: dict=None, curv_kwargs: dict=None,
                 acc_steps=1, update_ema=False, update_inv=False):

        if curv_shapes is None:
            self.curv_shapes = {}

        self.update_ema = update_ema
        self.update_inv = update_inv

        self.acc_steps = acc_steps
        self._count = 0

        self.layers = []

        for name, module in model.named_modules():
            if len(list(module.children())) > 0:
                continue

            module_name = module.__class__.__name__
            curv_shape = curv_shapes.get(module_name, '')
            curv_class = get_curv_class(module, curv_type, curv_shape)
            if curv_class is None:
                continue
            if curv_kwargs is None:
                curv_kwargs = {}
            curvature = curv_class(module, **curv_kwargs)
            layer = Layer(module, name, curvature)
            self.layers.append(layer)

    @property
    def local_layers(self):
        return self.layers

    def step(self):
        acc_steps = self.acc_steps

        # accumulatejG
        for layer in self.layers:
            curv = layer.curv
            if curv is not None:
                layer.acc_curv.update(curv.data, scale=1/acc_steps)

        # update acc step
        self._count += 1

        if self._count < acc_steps:
            return

        self._count = 0

        self.accumulate_postprocess()

        for layer in self.local_layers:
            curv = layer.curv
            if self.update_ema:
                curv.update_ema()
            if self.update_inv:
                curv.update_inv()

    def accumulate_postprocess(self):
        for layer in self.layers:
            layer.curv.data = layer.acc_curv.get()

    def get_eigenvalues(self):
        eigenvalues = []
        for layer in self.layers:
            eigenvalues.append(layer.curv.get_eigenvalues())

        eigenvalues = torch.cat(eigenvalues)
        sorted_e, _ = torch.sort(eigenvalues)

        return sorted_e

