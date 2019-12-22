import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsso.autograd import save_batched_grads


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
        if n == 1:
            x = torch.cat((x, x), 0)
        return x.double()

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
        if n == 1:
            x = torch.cat((x, x), 0)
        return x.double()


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
        return x.double()

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
        return x.double()


class ConvNet3D(ConvNet1D):

    def __init__(self):
        super().__init__()
        self.downsample = nn.Conv3d(3, 5, 3, stride=2, padding=1)
        self.upsample = nn.ConvTranspose3d(5, 3, 3, stride=2, padding=1)

    @staticmethod
    def get_random_input(n=1):
        c, t, h, w = 3, 12, 12, 12
        x = torch.randn(n, c, t, h, w)
        return x.double()


class BertEmbeddings(nn.Module):

    def __init__(self, vocab_size=30522, max_position_embeddings=512, type_vocab_size=2, hidden_size=10):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        return embeddings

    @staticmethod
    def get_random_input(n=1):
        seq_length = 128
        x = torch.randint(100, (n, seq_length))
        return x

    @staticmethod
    def get_loss(inputs, outputs):
        targets = torch.randn(outputs.size())
        error = targets - outputs
        loss = torch.sum(error * error) / 2 / len(inputs)
        return loss


def test_batched_grads(arch_cls, thr=1e-7):
    n = 2
    model = arch_cls()
    x = arch_cls.get_random_input(n=n)

    model = model.double()

    with save_batched_grads(model):
        output = model(x)
        loss = arch_cls.get_loss(x, output)
        loss.backward()

    for module in model.children():
        for p in module.parameters():
            if p.requires_grad:
                error = p.grads.view(-1, *p.grad.size()).sum(0) - p.grad
                assert error.norm() < thr, f'Error is too large for {module}' \
                                             f' param_size={p.size()}' \
                                             f' error_norm={error.norm()}.'


if __name__ == '__main__':
    test_batched_grads(LeNetBatchNorm)
    test_batched_grads(LeNetBatchNorm3D)
    test_batched_grads(ConvNet1D)
    test_batched_grads(ConvNet2D)
    test_batched_grads(ConvNet3D)
    test_batched_grads(BertEmbeddings)
