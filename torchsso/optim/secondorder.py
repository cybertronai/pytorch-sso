from collections import defaultdict
import math

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Optimizer
import torchsso
from torchsso.utils import TensorAccumulator
from torchsso.utils.chainer_communicators import create_communicator
from torchsso.utils.chainer_communicators import _utility


class SecondOrderOptimizer(Optimizer):
    r"""An optimizer for Second-Order Optimization.

    This optimizer manages the curvatures for each layer as a collection
        of torchsso.Curvature instance.
    This optimizer updates the params with the gradients pre-conditioned
        by the inverse of the curvature for each layer.

    Args:
        model (torch.nn.Module): model with parameters to be trained
        curv_type (str): type of the curvature ('Hessian', 'Fisher', or 'Cov')
        curv_shapes (dict): shape the curvatures for each type of layer
        curv_kwargs (dict): arguments (with keys) to be passed to torchsso.Curvature.__init__()
        lr (float, optional): learning rate
        momentum (float, optional): momentum factor
        momentum_type (str, optional): type of gradients of which momentum
            is calculated ('raw' or 'preconditioned')
        grad_ema_decay (float, optional): decay rate for EMA of gradients
        grad_ema_type (str, optional): type of gradients of which EMA
            is calculated ('raw' or 'preconditioned')
        l2_reg (float, optional): L2 penalty
        weight_decay (float, optional): weight decay
        normalizing_weights (bool, optional): whether the scale of the params
            are normalized after each step
        weight_scale (float, optional): the scale of the params for normalizing weights
        acc_steps (int, optional): number of steps for which gradients and curvatures
            are accumulated before each step
        non_reg_for_bn (bool, optional): whether the regularization is applied to BatchNorm params
        bias_correction (bool, optional): whether the bias correction (refer torch.optim.Adam) is applied
        lars (bool, optional): whether LARS (https://arxiv.org/abs/1708.03888) is applied
        lars_type (str, optional): type of gradients of which LARS
            is applied ('raw' or 'preconditioned')
        update_inv (bool, optional): whether to update curvature inverses at each step
        precondition_grad (bool, optional): whether to apply preconditioning
            (if False, this optimizer works as SGD)

    Example:
        >>> curv_shapes = {"Conv2d": "Kron", "Linear": "Diag"}
        >>> curv_kwargs = {"damping": 1e-3, "ema_decay": 0.999}
        >>> optimizer = torchsso.optim.SecondOrderOptimizer(model, "Cov", curv_shapes, curv_kwargs)
        >>>
        >>> def closure():
        >>>    optimizer.zero_grad()
        >>>    output = model(data)
        >>>    loss = F.cross_entropy(output, target)
        >>>    loss.backward(create_graph=args.create_graph)
        >>>    return loss, output
        >>>
        >>> optimizer.step(closure=closure)
    """

    def __init__(self, model: nn.Module, curv_type: str, curv_shapes: dict, curv_kwargs: dict,
                 lr=0.01, momentum=0., momentum_type='preconditioned',
                 grad_ema_decay=1., grad_ema_type='raw', l2_reg=0., weight_decay=0.,
                 normalizing_weights=False, weight_scale=None,
                 acc_steps=1, non_reg_for_bn=False, bias_correction=False,
                 lars=False, lars_type='preconditioned', update_inv=True, precondition_grad=True):

        if lr < 0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0:
            raise ValueError("Invalid momentum: {}".format(momentum))
        if momentum > 0 and momentum_type not in ['raw', 'preconditioned']:
            raise ValueError("Invalid momentum type: {}".format(momentum_type))
        if grad_ema_decay < 0 or 1 < grad_ema_decay:
            raise ValueError("Invalid grad_ema value: {}".format(grad_ema_decay))
        if grad_ema_decay > 0 and grad_ema_type not in ['raw', 'preconditioned']:
            raise ValueError("Invalid grad_ema type: {}".format(grad_ema_type))
        if l2_reg < 0:
            raise ValueError("Invalid l2_reg value: {}".format(l2_reg))
        if weight_decay < 0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if acc_steps < 1:
            raise ValueError("Invalid acc_steps: {}".format(acc_steps))
        if lars and lars_type not in ['raw', 'preconditioned']:
            raise ValueError("Invalid LARS type: {}".format(lars_type))
        if normalizing_weights and weight_scale is not None and weight_scale <= 0:
            raise ValueError("Invalid weight scale for LARS: {}".format(weight_scale))

        self.model = model
        defaults = {'lr': lr, 'momentum': momentum, 'momentum_type': momentum_type,
                    'grad_ema_decay': grad_ema_decay, 'grad_ema_type': grad_ema_type,
                    'l2_reg': l2_reg, 'weight_decay': weight_decay,
                    'normalizing_weights': normalizing_weights, 'weight_scale': weight_scale,
                    'acc_steps': acc_steps, 'bias_correction': bias_correction,
                    'lars': lars, 'lars_type': lars_type}
        defaults.update(curv_kwargs)
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.optim_state = {'step': 0, 'acc_step': 0}

        self.param_groups = []
        self.curv_type = curv_type
        self.curv_shapes = {} if curv_shapes is None else curv_shapes
        self.update_inv = update_inv
        self.precondition_grad = precondition_grad

        for module in model.modules():
            if len(list(module.children())) > 0:
                continue
            params = list(module.parameters())
            if len(params) == 0:
                continue

            curv_class = self.get_curv_class(module)
            curvature = curv_class(module, **curv_kwargs)

            group = {
                'params': params,
                'curv': curvature,
                'acc_curv': TensorAccumulator(),
                'acc_grads': TensorAccumulator()
            }

            self.add_param_group(group)
            self.init_buffer(params)

            if non_reg_for_bn and \
                    isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                group['l2_reg'] = 0
                group['weight_decay'] = 0
                group['normalizing_weights'] = False

    def init_buffer(self, params):
        for p in params:
            state = self.state[p]
            state['momentum_buffer'] = torch.zeros_like(p.data)
            state['grad_ema_buffer'] = torch.zeros_like(p.data)

    @property
    def local_param_groups(self):
        return self.param_groups

    def get_curv_class(self, module):
        module_name = module.__class__.__name__
        curv_shape = self.curv_shapes.get(module_name, '')
        curv_name = curv_shape + self.curv_type + module_name
        curv_class = getattr(torchsso, curv_name, None)

        assert curv_class is not None, f"Failed to lookup Curvature class {curv_name} for {module}."

        return curv_class

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        n = self.defaults['acc_steps']
        loss = None

        if closure is not None:
            # forward and backward
            loss = closure()

            # accumulate
            for group in self.param_groups:
                params = group['params']

                grads = [p.grad.data for p in params]
                group['acc_grads'].update(grads, scale=1/n)

                curv = group['curv']
                if curv is not None:
                    group['acc_curv'].update(curv.data, scale=1/n)

            # update acc step
            self.optim_state['acc_step'] += 1
            if self.optim_state['acc_step'] < n:
                return loss
            else:
                self.optim_state['acc_step'] = 0

            self.backward_postprocess()

        self.optim_state['step'] += 1

        for group in self.local_param_groups:

            self.update_preprocess(group, grad_type='raw')

            # update curvature
            params, curv = group['params'], group['curv']
            if curv is not None:
                curv.step(update_inv=self.update_inv)
                if self.precondition_grad:
                    curv.precondition_grad(params)

            # update params
            self.update_preprocess(group, grad_type='preconditioned')
            self.update(group)
            self.update_postprocess(group)

        return loss

    def backward_postprocess(self, target='params'):
        for group in self.param_groups:
            params = group[target]

            acc_grads = group['acc_grads'].get()
            for p, acc_grad in zip(params, acc_grads):
                p.grad = acc_grad.clone()

            curv = group['curv']
            if curv is not None:
                curv.data = group['acc_curv'].get()

    def update(self, group, target='params'):
        params = group[target]
        for p in params:
            grad = p.grad
            if grad is None:
                continue
            p.data.add_(-group['lr'], grad)

    def update_preprocess(self, group, target='params', grad_type='raw'):
        assert grad_type in ['raw', 'preconditioned'], 'Invalid grad type: {}.'.format(grad_type)
        params = group[target]
        state = self.state

        def apply_l2_reg(p, grad):
            if group['l2_reg'] != 0:
                if grad.is_sparse:
                    raise RuntimeError(
                        "l2 regularization option is not compatible with sparse gradients")
                grad.add_(group['l2_reg'], p.data)
                curv = group['curv']
                if curv is not None:
                    curv.l2_reg = group['l2_reg']

        def apply_weight_decay(p, grad):
            if group['weight_decay'] != 0:
                if hasattr(grad, 'is_sparse') and grad.is_sparse:
                    raise RuntimeError(
                        "weight_decay option is not compatible with sparse gradients")
                grad.add_(group['weight_decay'], p.data)

        def apply_momentum(p, grad):
            momentum = group['momentum']

            if momentum != 0:
                buf = state[p]['momentum_buffer']
                buf.mul_(momentum).add_(grad)
                grad.copy_(buf)

        def apply_grad_ema_decay(p, grad):
            grad_ema_decay = group['grad_ema_decay']
            if grad_ema_decay != 1:
                buf = state[p]['grad_ema_buffer']
                buf.mul_(1 - grad_ema_decay).add_(grad.mul(grad_ema_decay))
                grad.copy_(buf)

        def apply_bias_correction(grad):
            curv = group['curv']
            beta1 = 1 - group['grad_ema_decay']
            beta2 = 1 - curv.ema_decay

            bias_correction1 = 1 - beta1 ** self.optim_state['step']
            bias_correction2 = 1 - beta2 ** self.optim_state['step']
            if getattr(curv, 'use_sqrt_ema', False):
                bias_correction2 = math.sqrt(bias_correction2)

            grad.mul_(bias_correction2 / bias_correction1)

        def apply_lars(p, grad, thr=1e-2, eps=1e-9):
            d_norm = p.data.norm()
            if d_norm > thr:
                g_norm = grad.norm()
                rate = d_norm / (g_norm + eps)
                grad.mul_(rate)

        for p in params:

            grad = p.grad

            if grad is None:
                continue

            if grad_type == 'raw':
                apply_l2_reg(p, grad)

            if grad_type == 'preconditioned':
                apply_weight_decay(p, grad)

            if group['momentum_type'] == grad_type:
                apply_momentum(p, grad)

            if group['grad_ema_type'] == grad_type:
                apply_grad_ema_decay(p, grad)

            if grad_type == 'preconditioned' and group['bias_correction']:
                apply_bias_correction(grad)

            if group['lars_type'] == grad_type and group['lars']:
                apply_lars(p, grad)

    def update_postprocess(self, group, target='params'):
        params = group[target]
        curv = group['curv']

        def apply_normalizing_weights(p, thr=1e-2, eps=1e-9):
            d_norm = p.data.norm()
            if d_norm > thr:
                scale = group['weight_scale']
                if scale is None:
                    scale = np.sqrt(2.0 * w.data.shape[0])
                p.data.div_(d_norm + eps).mul_(scale)

        if group['normalizing_weights']:
            for p, _p in zip(params, group['params']):
                w = getattr(curv.module, 'weight', None)
                if w is not None and w is _p:
                    apply_normalizing_weights(p)


class DistributedSecondOrderOptimizer(SecondOrderOptimizer):

    def __init__(self, *args, **kwargs):

        self.actual_optimizer.__init__(self, *args, **kwargs)

        self.comm = create_communicator()

        local_size = self.comm.size
        local_rank = self.comm.rank
        indices = np.array_split(np.arange(len(self.param_groups)), local_size)
        indices = [local_indices.tolist() for local_indices in indices]
        local_indices = indices[local_rank]
        local_param_groups = [self.param_groups[i] for i in local_indices]

        self.indices = indices
        self.local_indices = local_indices
        self._local_param_groups = local_param_groups
        setattr(self.comm, 'indices', indices)

    @property
    def actual_optimizer(self):
        return SecondOrderOptimizer

    @property
    def local_param_groups(self):
        return self._local_param_groups

    def extractors_for_rsv(self):
        extractors = [_utility.extract_attr_from_params('grad'),
                      _utility.extract_attr_from_curv('data', True)]
        return extractors

    def extractors_for_agv(self):
        extractors = [_utility.extract_attr_from_params('data')]
        return extractors

    def backward_postprocess(self, target='params'):
        self.actual_optimizer.backward_postprocess(self, target)
        # reduce_scatter_v
        self.comm.reduce_scatterv_data(self.param_groups, self.extractors_for_rsv())

    def is_updated(self):
        return self.optim_state['acc_step'] == 0

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        ret = self.actual_optimizer.step(self, closure)

        if self.is_updated():
            # all_gather_v
            self.comm.allgatherv_data(self.param_groups, self.extractors_for_agv())

        return ret

