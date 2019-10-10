import torch
import torch.nn.functional as F

from torchsso.utils import TensorAccumulator


class Fisher(object):

    def __init__(self):
        self.prob = None
        self._do_backward = True
        self._acc_cov = TensorAccumulator()

    @property
    def do_backward(self):
        return self._do_backward

    def turn_on_backward(self):
        self._do_backward = True

    def turn_off_backward(self):
        self._do_backward = False

    def accumulate_cov(self, cov):
        self._acc_cov.update(cov)

    def finalize(self):
        return self._acc_cov.get()

    def update_as_presoftmax(self, prob):
        raise NotImplementedError('This method supports only torchsso.KronFisherLinear.')

        
def get_closure_for_fisher(optimizer, model, data, target, approx_type=None, num_mc=1):

    _APPROX_TYPE_MC = 'mc'

    def turn_off_param_grad():
        for group in optimizer.param_groups:
            group['curv'].turn_on_backward()
            for param in group['params']:
                param.requires_grad = False

    def turn_on_param_grad():
        for group in optimizer.param_groups:
            group['curv'].turn_off_backward()
            for param in group['params']:
                param.requires_grad = True

    def closure():

        for group in optimizer.param_groups:
            assert isinstance(group['curv'], Fisher), f"Invalid Curvature type: {type(group['curv'])}."

        optimizer.zero_grad()
        output = model(data)
        prob = F.softmax(output, dim=1)

        is_sampling = approx_type is None or approx_type == _APPROX_TYPE_MC

        if is_sampling:
            turn_off_param_grad()

            if approx_type == _APPROX_TYPE_MC:
                dist = torch.distributions.Categorical(prob)
                _target = dist.sample((num_mc,))
                for group in optimizer.param_groups:
                    group['curv'].prob = torch.ones_like(prob[:, 0]).div(num_mc)

                for i in range(num_mc):
                    loss = F.cross_entropy(output, _target[i])
                    loss.backward(retain_graph=True)
            else:
                for i in range(model.num_classes):
                    for group in optimizer.param_groups:
                        group['curv'].prob = prob[:, i]
                    loss = F.cross_entropy(output, torch.ones_like(target).mul(i))
                    loss.backward(retain_graph=True)

            turn_on_param_grad()

        else:
            raise ValueError('Invalid approx type: {}'.format(approx_type))

        loss = F.cross_entropy(output, target)
        loss.backward()

        return loss, output

    return closure
