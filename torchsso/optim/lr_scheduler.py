from torch.optim import Optimizer


class _IterLRScheduler(object):
    def __init__(self, optimizer, last_iter=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_iter + 1)
        self.last_iter = last_iter
        self.scheduler_type = 'iter'

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, iter=None):
        if iter is None:
            iter = self.last_iter + 1
        self.last_iter = iter
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class PolynomialDecayIterLR(_IterLRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every iter. When last_iter=-1, sets initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_iter (int): The index of last iter. Default: -1.
    """

    def __init__(self, optimizer, rate, max_count, target=None, start_iter=0, last_iter=-1):
        self.rate = rate
        self.max_count = max_count
        self.target = target
        self.start_iter = start_iter
        super(PolynomialDecayIterLR, self).__init__(optimizer, last_iter)

    def get_lr(self):
        if self.last_iter < self.start_iter:
            return [param_group['lr']
                    for param_group in self.optimizer.param_groups]
        decay = max(1-(self.last_iter-self.start_iter) / (self.max_count-self.start_iter), 0)
        if self.target is not None:
            if self.rate > 0:
                return [self.target if self.target / (base_lr * decay ** self.rate) > 1
                        else base_lr * decay ** self.rate
                        for base_lr in self.base_lrs]
            else:
                return [self.target if self.target / (base_lr * decay ** self.rate) < 1
                        else base_lr * decay ** self.rate
                        for base_lr in self.base_lrs]
        return [base_lr * decay ** self.rate
                for base_lr in self.base_lrs]


class GradualWarmupIterLR(_IterLRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every iter. When last_iter=-1, sets initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_iter (int): The index of last iter. Default: -1.
    """

    def __init__(self, optimizer, initial_lr, max_count, last_iter=-1):
        self.initial_lr = initial_lr
        self.max_count = max_count
        super(GradualWarmupIterLR, self).__init__(optimizer, last_iter)

    def get_lr(self):
        if self.last_iter > self.max_count:
            return [param_group['lr']
                    for param_group in self.optimizer.param_groups]
        else:
            alpha = self.last_iter / self.max_count
            return [self.initial_lr*(1-alpha) + base_lr*alpha
                    for base_lr in self.base_lrs]


class MomentumCorrectionLR(object):

    def __init__(self, scheduler):
        super(MomentumCorrectionLR, self).__setattr__(
            'scheduler', scheduler)

        for group in self.optimizer.param_groups:
            group['init_momentum'] = group['momentum']

    def step(self, count=None):
        self.scheduler.step(count)

        for group in self.optimizer.param_groups:
            lr = group['lr']
            lr_pre = group.get('lr_pre', None)

            if lr_pre is not None:
                m = group.get('init_momentum', 0)
                group['momentum'] = m * lr / lr_pre

            group['lr_pre'] = group['lr']

    def __getattr__(self, item):
        return getattr(self.scheduler, item)

    def __setattr__(self, key, value):
        setattr(self.scheduler, key, value)
