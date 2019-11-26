import torch
import torch.nn as nn
import torch.nn.functional as F

import torchsso


class LeNetBatchNorm(nn.Module):

    def __init__(self, num_classes=10, affine=True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6, affine=affine)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16, affine=affine)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(120, affine=affine)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84, affine=affine)
        self.fc3 = nn.Linear(84, num_classes, bias=False)
        self.pool_func = F.max_pool2d

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.pool_func(h, 2)
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.pool_func(h, 2)
        h = h.view(h.size(0), -1)
        h = F.relu(self.bn3(self.fc1(h)))
        h = F.relu(self.bn4(self.fc2(h)))
        output = self.fc3(h)
        return output

    @staticmethod
    def get_random_input(n=1):
        c, h, w = 3, 32, 32
        x = torch.randn(n, c, h, w)
        return x

    @staticmethod
    def get_loss(inputs, outputs):
        targets = torch.randn(outputs.size())
        error = targets - outputs
        loss = torch.sum(error * error) / 2 / len(inputs)
        return loss


class LeNetBatchNorm3D(LeNetBatchNorm):

    def __init__(self, num_classes=10, affine=True):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 6, 5)
        self.bn1 = nn.BatchNorm3d(6, affine=affine)
        self.conv2 = nn.Conv3d(6, 16, 5)
        self.bn2 = nn.BatchNorm3d(16, affine=affine)
        self.fc1 = nn.Linear(16 * 5 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(120, affine=affine)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84, affine=affine)
        self.fc3 = nn.Linear(84, num_classes, bias=False)
        self.pool_func = F.max_pool3d

    @staticmethod
    def get_random_input(n=1):
        c, t, h, w = 3, 32, 32, 32
        x = torch.randn(n, c, t, h, w)
        return x


class ConvNet1D(nn.Module):

    def __init__(self):
        super().__init__()
        self.downsample = nn.Conv1d(3, 5, 3, stride=2, padding=1, bias=True)
        self.upsample = nn.ConvTranspose1d(5, 3, 3, stride=2, padding=1, bias=True)

    def forward(self, x):
        h = self.downsample(x)
        output = self.upsample(h, output_size=x.size())
        return output

    @staticmethod
    def get_random_input(n=1):
        c, l = 3, 12
        x = torch.randn(n, c, l)
        return x

    @staticmethod
    def get_loss(inputs, outputs):
        error = outputs - inputs
        loss = torch.sum(error * error) / 2 / len(inputs)
        return loss


class ConvNet2D(ConvNet1D):

    def __init__(self):
        super().__init__()
        self.downsample = nn.Conv2d(3, 5, 3, stride=2, padding=1, bias=True)
        self.upsample = nn.ConvTranspose2d(5, 3, 3, stride=2, padding=1, bias=True)

    @staticmethod
    def get_random_input(n=1):
        c, h, w = 3, 12, 12
        x = torch.randn(n, c, h, w)
        return x


class ConvNet3D(ConvNet1D):

    def __init__(self):
        super().__init__()
        self.downsample = nn.Conv3d(3, 5, 3, stride=2, padding=1, bias=True)
        self.upsample = nn.ConvTranspose3d(5, 3, 3, stride=2, padding=1, bias=True)

    @staticmethod
    def get_random_input(n=1):
        c, t, h, w = 3, 12, 12, 12
        x = torch.randn(n, c, t, h, w)
        return x


def _step(arch_cls, curv_shapes):
    n = 10
    model = arch_cls()
    x = arch_cls.get_random_input(n=n)

    curv_kwargs = {"damping": 1e-3, "ema_decay": 0.999}
    optimizer = torchsso.optim.SecondOrderOptimizer(model, 'Cov', curv_shapes, curv_kwargs)

    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = arch_cls.get_loss(x, output)
        loss.backward()

        return loss

    optimizer.step(closure=closure)


def test_diag_curv(arch_cls):
    curv_shapes = {'Conv1d': 'Diag',
                   'Conv2d': 'Diag',
                   'Conv3d': 'Diag',
                   'ConvTranspose1d': 'Diag',
                   'ConvTranspose2d': 'Diag',
                   'ConvTranspose3d': 'Diag',
                   'BatchNorm1d': 'Diag',
                   'BatchNorm2d': 'Diag',
                   'BatchNorm3d': 'Diag',
                   'Linear': 'Diag'}

    _step(arch_cls, curv_shapes)


def test_kron_curv(arch_cls):
    curv_shapes = {'Conv1d': 'Kron',
                   'Conv2d': 'Kron',
                   'Conv3d': 'Kron',
                   'ConvTranspose1d': 'Kron',
                   'ConvTranspose2d': 'Kron',
                   'ConvTranspose3d': 'Kron',
                   'Linear': 'Kron'}

    _step(arch_cls, curv_shapes)


if __name__ == '__main__':
    test_diag_curv(LeNetBatchNorm)
    test_diag_curv(LeNetBatchNorm3D)
    test_diag_curv(ConvNet1D)
    test_diag_curv(ConvNet2D)
    test_diag_curv(ConvNet3D)
    test_kron_curv(ConvNet1D)
    test_kron_curv(ConvNet2D)
    test_kron_curv(ConvNet3D)
