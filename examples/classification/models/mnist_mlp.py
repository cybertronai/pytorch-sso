import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['mlp']


class MNIST_MLP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        n_hid = 2
        n_out = 10
        self.l1 = nn.Linear(28*28, n_hid)
        self.l2 = nn.Linear(n_hid, n_hid)
        self.l3 = nn.Linear(n_hid, n_out)

    def forward(self, x: torch.Tensor):
        x1 = x.view([-1, 28*28])
        x2 = F.relu(self.l1(x1))
        x3 = F.relu(self.l2(x2))
        x4 = self.l3(x3)
        return x4


def mlp(**kwargs):
    model = MNIST_MLP(**kwargs)
    return model

