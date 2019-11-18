import math

import torch
import torch.nn as nn
import torchsso

PI_TYPE_TRACENORM = 'tracenorm'


class Curvature(object):
    r"""Base implementation of the curvatures for each layer.

    This class computes/maintains curvature (data) and EMA/inverse of it for a given layer (module)
        which are used for torchsso.optim.SecondOrderOptimizer.
    Standard deviation (std) is calculated for torchsso.optim.VIOptimizer based on the inverse.
    IE, data -> ema -> inv (-> std)

    Args:
        module (torch.nn.Module): a layer with trainable params for which the curvature is computed
        ema_decay (float, optional): decay rate for EMA of curvature
        damping (float, optional): value to be added to the diagonal of EMA before inverting it
        use_max_ema (bool, optional): whether to use the maximum value as EMA
        use_sqrt_ema (bool, optional): whether to take the squre root of EMA
    """

    def __init__(self, module: nn.Module, ema_decay=1., damping=1e-7,
                 use_max_ema=False, use_sqrt_ema=False,
                 pi_type=PI_TYPE_TRACENORM):

        if ema_decay < 0 or 1 < ema_decay:
            raise ValueError("Invalid ema_decay: {}".format(ema_decay))
        if damping < 0:
            raise ValueError("Invalid damping: {}".format(damping))
        if pi_type not in [PI_TYPE_TRACENORM]:
            raise ValueError("Invalid pi_type: {}".format(pi_type))

        self._module = module
        self.ema_decay = ema_decay
        self._damping = damping
        self._l2_reg = 0
        self._l2_reg_ema = 0

        self._data = None
        self._acc_data = None
        self.ema = None
        self.ema_max = None
        self.inv = None
        self.std = None

        self.use_sqrt_ema = use_sqrt_ema
        self.use_max_ema = use_max_ema

        self.pi_type = pi_type

        module.register_forward_hook(self.forward_postprocess)
        module.register_backward_hook(self.backward_postprocess)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def shape(self):
        if self._data is None:
            return self._get_shape()

        return tuple([d.shape for d in self._data])

    @property
    def device(self):
        return next(self._module.parameters()).device

    def _get_shape(self):
        size = 0
        for p in self._module.parameters():
            size += p.view(-1).shape[0]

        return tuple((size, size))

    def element_wise_init(self, value):
        init_data = []
        for s in self.shape:
            diag = torch.ones(s[0], device=self.device).mul(value)  # 1d
            diag = torch.diag(diag)  # 1d -> 2d
            init_data.append(diag)

        self._data = init_data

    @property
    def module(self):
        return self._module

    @property
    def bias(self):
        bias = getattr(self._module, 'bias', None)
        return False if bias is None else True

    @property
    def damping(self):
        return self._damping + self._l2_reg_ema

    @property
    def l2_reg(self):
        return self._l2_reg

    @l2_reg.setter
    def l2_reg(self, value):
        self._l2_reg = value

    @property
    def l2_reg_ema(self):
        return self._l2_reg_ema

    def forward_postprocess(self, module, input, output):
        assert self._module == module

        data_input = input[0].detach()

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
        setattr(module, 'data_output', output)

        self.update_in_forward(data_input)

    def backward_postprocess(self, module, grad_input, grad_output):
        assert self._module == module

        index = 1 if self.bias else 0
        grad_input = None if grad_input[index] is None else grad_input[index].detach()
        grad_output = grad_output[0]

        setattr(module, 'grad_input', grad_input)
        setattr(module, 'grad_output', grad_output)

        self.update_in_backward(grad_output)

        # adjust grad scale along with 'reduction' in loss function
        batch_size = grad_output.shape[0]
        self.adjust_data_scale(batch_size**2)

    def adjust_data_scale(self, scale):
        self._data = [d.mul(scale) for d in self._data]

    def update_in_forward(self, data_input):
        pass

    def update_in_backward(self, grad_output):
        raise NotImplementedError

    def step(self, update_std=False, update_inv=True):
        # TODO(oosawak): Add check for ema/inv timing
        self.update_ema()
        if update_inv:
            self.update_inv()
        if update_std:
            self.update_std()

    def update_ema(self):
        data = self.data
        ema = self.ema
        ema_max = self.ema_max
        beta = self.ema_decay
        if ema is None or beta == 1:
            self.ema = [d.clone() for d in data]
            if self.use_max_ema and ema_max is None:
                self.ema_max = [e.clone() for e in self.ema]
            self._l2_reg_ema = self._l2_reg
        else:
            self.ema = [d.mul(beta).add(1 - beta, e)
                        for d, e in zip(data, ema)]
            self._l2_reg_ema = self._l2_reg * beta + self._l2_reg_ema * (1 - beta)

        if self.use_max_ema:
            for e, e_max in zip(self.ema, self.ema_max):
                torch.max(e, e_max, out=e_max)

    def update_inv(self):
        ema = self.ema if not self.use_max_ema else self.ema_max
        self.inv = [self._inv(e) for e in ema]

    def _inv(self, X):
        X_damp = add_value_to_diagonal(X, self.damping)

        return torchsso.utils.inv(X_damp)

    def precondition_grad(self, params):
        raise NotImplementedError

    def update_std(self):
        raise NotImplementedError

    def sample_params(self, params, mean, std_scale):
        raise NotImplementedError

    def std_norm(self):
        raise NotImplementedError


class DiagCurvature(Curvature):

    def _get_shape(self):
        return tuple(p.shape for p in self.module.parameters())

    def element_wise_init(self, value):
        self._data = [torch.ones(s, device=self.device).mul(value) for s in self.shape]

    def update_in_backward(self, grad_output_data):
        raise NotImplementedError

    def _inv(self, X):
        if self.use_sqrt_ema:
            X = X.sqrt()

        X_damp = X.add(X.new_ones(X.shape).mul(self.damping))

        return 1 / X_damp

    def precondition_grad(self, params):
        for p, inv in zip(params, self.inv):
            preconditioned_grad = inv.mul(p.grad)

            p.grad.copy_(preconditioned_grad)

    def update_std(self):
        self.std = [inv.sqrt() for inv in self.inv]

    def sample_params(self, params, mean, std_scale):
        for p, m, std in zip(params, mean, self.std):
            noise = torch.randn_like(m)
            p.data.copy_(torch.addcmul(m, std_scale, noise, std))

    def std_norm(self):
        if self.std is None:
            return 0

        return sum(std.norm().item() for std in self.std)


class KronCurvature(Curvature):

    def __init__(self, *args, **kwargs):
        super(KronCurvature, self).__init__(*args, **kwargs)

        self._A = None
        self._G = None

    @property
    def data(self):
        return [self._A, self._G]

    @data.setter
    def data(self, value):
        self._A, self._G = value

    @property
    def shape(self):
        if self._A is None or self._G is None:
            return self._get_shape()

        return self._A.shape, self._G.shape

    def _get_shape(self):
        raise NotImplementedError

    def element_wise_init(self, value):
        super(KronCurvature, self).element_wise_init(math.sqrt(value))
        self._A, self._G = self._data

    @property
    def A(self):
        return self._A

    @property
    def G(self):
        return self._G

    def update_in_forward(self, input_data):
        raise NotImplementedError

    def update_in_backward(self, grad_output_data):
        raise NotImplementedError

    def adjust_data_scale(self, scale):
        self._G.mul_(scale)

    def update_inv(self):
        A, G = self.ema

        if self.pi_type == PI_TYPE_TRACENORM:
            pi = torch.sqrt((A.trace()/A.shape[0])/(G.trace()/G.shape[0]))
        else:
            pi = 1.

        r = self.damping**0.5
        self.inv = [torchsso.utils.inv(add_value_to_diagonal(X, value))
                    for X, value in zip([A, G], [r*pi, r/pi])]

    def precondition_grad(self, params):
        raise NotImplementedError

    def update_std(self):
        A_inv, G_inv = self.inv

        self.std = [torchsso.utils.cholesky(X)
                    for X in [A_inv, G_inv]]

    def sample_params(self, params, mean, std_scale):
        raise NotImplementedError

    def std_norm(self):
        if self.std is None:
            return 0

        A_ic, G_ic = self.std
        return A_ic.norm().item() * G_ic.norm().item()


def add_value_to_diagonal(X, value):
    if torch.cuda.is_available():
        indices = torch.cuda.LongTensor([[i, i] for i in range(X.shape[0])])
    else:
        indices = torch.LongTensor([[i, i] for i in range(X.shape[0])])
    values = X.new_ones(X.shape[0]).mul(value)
    return X.index_put(tuple(indices.t()), values, accumulate=True)
