import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsso.autograd import save_sample_grads


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
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.bn3(self.fc1(out)))
        out = F.relu(self.bn4(self.fc2(out)))
        out = self.fc3(out)
        return out


def test_samplegrad():
    model = LeNet5BatchNorm()
    n = 10
    c, h, w = 3, 32, 32
    x = torch.randn(n, c, h, w)

    with save_sample_grads(model):
        out = model(x)
        loss = out.sum()
        loss.backward()

    for module in model.children():
        print(module)
        for p in module.parameters():
            if p.requires_grad:
                error = (p.grads.sum(0) - p.grad).max()
                print(f'\t{p.size()} : error = {error}')


if __name__ == '__main__':
    test_samplegrad()
