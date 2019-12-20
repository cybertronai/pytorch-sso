import torch.nn as nn
import torchsso
from torchsso.curv.curvature import Curvature, DiagCurvature, KronCurvature  # NOQA
from torchsso.curv.cov.linear import CovLinear, DiagCovLinear, KronCovLinear  # NOQA
from torchsso.curv.cov.conv import DiagCovConvNd, DiagCovConvTransposeNd, KronCovConvNd, KronCovConvTransposeNd  # NOQA
from torchsso.curv.cov.batchnorm import DiagCovBatchNormNd  # NOQA


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

    assert curv_class is not None, f"Failed to lookup Curvature class {curv_name} for {module}."

    return curv_class
