from contextlib import contextmanager
import numpy as np
import torch

REQUIRES_GRAD_ATTR = '_original_requires_grad'


def requires_grad(module, param_name):
    param = getattr(module, param_name, None)
    return param is not None and getattr(param, REQUIRES_GRAD_ATTR)


@contextmanager
def disable_param_grad(model):

    for param in model.parameters():
        setattr(param, REQUIRES_GRAD_ATTR, param.requires_grad)
        param.requires_grad = False

    yield
    for param in model.parameters():
        param.requires_grad = getattr(param, REQUIRES_GRAD_ATTR)


def matrix_to_tril(mat: torch.Tensor):
    """
    Convert matrix (2D array)
    to lower triangular of it (1D array, row direction)

    Example:
      [[1, x, x],
       [2, 3, x], -> [1, 2, 3, 4, 5, 6]
       [4, 5, 6]]
    """
    assert mat.ndim == 2
    tril_indices = torch.tril_indices(*mat.shape)
    return mat[tril_indices[0], tril_indices[1]]


def tril_to_matrix(tril: torch.Tensor):
    """
    Convert lower triangular of matrix (1D array)
    to full symmetric matrix (2D array)

    Example:
                            [[1, 2, 4],
      [1, 2, 3, 4, 5, 6] ->  [2, 3, 5],
                             [4, 5, 6]]
    """
    assert tril.ndim == 1
    n_cols = get_n_cols_by_tril(tril)
    rst = torch.zeros(n_cols, n_cols, device=tril.device, dtype=tril.dtype)
    tril_indices = torch.tril_indices(n_cols, n_cols)
    rst[tril_indices[0], tril_indices[1]] = tril
    rst = rst + rst.T - torch.diag(torch.diag(rst))
    return rst


def extend_A_tril(A_tril: torch.Tensor):
    """
    Extend A (input kronecker-factor) with ones
    in its lower triangular form.

    Example:
      original matrix:
      [[a, b],   [[a, b, 1],
       [c, d]] -> [c, d, 1],
                  [1, 1, 1]]

      lower triangular (row direction):
      [a, c, d] -> [a, c, d, 1, 1, 1]
    """
    assert A_tril.ndim == 1
    # extend
    n_cols = get_n_cols_by_tril(A_tril)
    ones = torch.ones(n_cols + 1, device=A_tril.device, dtype=A_tril.dtype)
    return torch.cat([A_tril, ones])


def get_n_cols_by_tril(tril:torch.Tensor):
    """
    Get number of columns of original matrix
    by lower triangular (tril) of it.

    ncols^2 + ncols = 2 * tril.numel()
    """
    assert tril.ndim == 1
    numel = tril.numel()
    return int(np.sqrt(2 * numel + 0.25) - 0.5)
