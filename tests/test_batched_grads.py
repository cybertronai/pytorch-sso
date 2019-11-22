import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsso.autograd import save_batched_grads


class LeNet5BatchNorm(nn.Module):
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

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.max_pool2d(h, 2)
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.max_pool2d(h, 2)
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


class ConvNet1D(nn.Module):

    def __init__(self):
        super().__init__()
        self.downsample = nn.Conv1d(3, 5, 3, stride=2, padding=1)
        self.upsample = nn.ConvTranspose1d(5, 3, 3, stride=2, padding=1)

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
        self.downsample = nn.Conv2d(3, 5, 3, stride=2, padding=1)
        self.upsample = nn.ConvTranspose2d(5, 3, 3, stride=2, padding=1)

    @staticmethod
    def get_random_input(n=1):
        c, h, w = 3, 12, 12
        x = torch.randn(n, c, h, w)
        return x


class ConvNet3D(ConvNet1D):

    def __init__(self):
        super().__init__()
        self.downsample = nn.Conv3d(3, 5, 3, stride=2, padding=1)
        self.upsample = nn.ConvTranspose3d(5, 3, 3, stride=2, padding=1)

    @staticmethod
    def get_random_input(n=1):
        c, t, h, w = 3, 12, 12, 12
        x = torch.randn(n, c, t, h, w)
        return x


def test_batched_grads(arch_cls, thr=1e-5):
    n = 10
    model = arch_cls()
    x = arch_cls.get_random_input(n=n)

    with save_batched_grads(model):
        output = model(x)
        loss = arch_cls.get_loss(x, output)
        loss.backward()

    for module in model.children():
        for p in module.parameters():
            if p.requires_grad:
                error = p.grads.sum(0) - p.grad
                ratio = error.norm() / (p.grad.norm() + 1.0)
                assert ratio < thr, f'Error is too large for {module}' \
                                    f' param_size={p.size()}' \
                                    f' error_ratio={ratio}.'


if __name__ == '__main__':
    test_batched_grads(LeNet5BatchNorm)
    test_batched_grads(ConvNet1D)
    test_batched_grads(ConvNet2D)
    test_batched_grads(ConvNet3D)
