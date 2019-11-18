from torch.optim import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class DistributedFirstOrderOptimizer(Optimizer):

    def __init__(self, optimizer, model, dist, lars=False):
        super(DistributedFirstOrderOptimizer, self).__setattr__(
           'actual_optimizer', optimizer
        )
        super(DistributedFirstOrderOptimizer, self).__setattr__(
            'model', model
        )
        super(DistributedFirstOrderOptimizer, self).__setattr__(
            'dist', dist
        )
        super(DistributedFirstOrderOptimizer, self).__setattr__(
            'lars', lars
        )

    def step(self, closure=None, thr=1e-2, eps=1e-9):
        loss = None
        if closure is not None:
            loss = closure()
            world_size = self.dist.get_world_size()
            grads = [p.grad for p in self.model.parameters()]
            # pack
            packed_tensor = parameters_to_vector(grads)
            # all reduce
            self.dist.all_reduce(packed_tensor)
            # unpack
            vector_to_parameters(packed_tensor.div_(world_size), grads)

        if self.lars:
            for group in self.param_groups:
                for p in group['params']:
                    setattr(p, 'data_pre', p.data.detach().clone())

        self.actual_optimizer.step(closure=None)

        if self.lars:
            for group in self.param_groups:
                for p in group['params']:
                    d_norm_pre = p.data_pre.norm()
                    if d_norm_pre > thr:
                        upd = p.data - p.data_pre
                        upd_norm = upd.norm()
                        rate = group['lr'] * d_norm_pre / (upd_norm + eps)
                        p.data = p.data_pre.add(rate, upd)

        return loss

    def __getattr__(self, item):
        return getattr(self.actual_optimizer, item)

    def __setattr__(self, key, value):
        if key == 'step':
            super().__setattr__(key, value)
        else:
            setattr(self.actual_optimizer, key, value)
