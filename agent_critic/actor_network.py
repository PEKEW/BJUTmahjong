import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch import nn


class ResnetBlockConv1d(nn.Module):
    """ 1D-Convolutional ResNet block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_h=None, size_out=None, res=True,
                 norm_method='batch_norm', kernel_size=(3,)):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        if norm_method == 'batch_norm':
            norm = nn.BatchNorm1d
        elif norm_method == 'sync_batch_norm':
            norm = nn.SyncBatchNorm
        else:
            norm = nn.Identity

        self.bn_0 = norm(size_in)
        self.bn_1 = norm(size_h)

        self.fc_0 = nn.Conv1d(size_in, size_h, kernel_size=kernel_size, padding=1)
        self.fc_1 = nn.Conv1d(size_h, size_out, kernel_size=kernel_size, padding=1)
        self.actvn = torch.relu
        self.res = res
        if size_in == size_out or res is False:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)

        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(self.bn_0(x)))
        dx = self.fc_1(self.actvn(self.bn_1(net)))
        if self.shortcut is not None:
            x_s = self.shortcut(x)
        elif self.res is False:
            x_s = 0.
        else:
            x_s = x
        out = x_s + dx
        return out


class playNet(nn.Module):
    def __init__(self, input_dim, num_block, hidden_dim):
        super(playNet, self).__init__()
        self.num_block = num_block = num_block
        self.hidden_dim = hidden_dim = hidden_dim
        self.input_dim = input_dim = input_dim
        self.transform = nn.ModuleList()
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=(3,), padding=1)
        for i in range(num_block):
            self.transform.append(ResnetBlockConv1d(hidden_dim, kernel_size=(3,)))
        self.last_conv = nn.Conv1d(hidden_dim, 1, kernel_size=(1,), padding=0)

    def forward(self, x, mask):
        # x [batch, D, 27]
        x = self.conv(x)
        for i in range(self.num_block):
            x = self.transform[i](x)
        x = self.last_conv(x)
        eps = torch.ones_like(x) * 1e-3
        x = torch.softmax(x, dim=2) + eps
        x = torch.squeeze(x)
        if x.dim() == 1:
            x = x[None, :]
        if mask.dim() == 1:
            mask = mask[None, :]
        x = x * mask  # NOTE
        x = x / torch.sum(x, dim=1, keepdim=True)
        assert x.sum() > x.shape[0] - 1e-2
        return x


class judgeNet(nn.Module):
    def __init__(self, input_dim, num_block, hidden_dim, out_dim=2):
        super(judgeNet, self).__init__()
        self.num_block = num_block = num_block
        self.hidden_dim = hidden_dim = hidden_dim
        self.input_dim = input_dim = input_dim
        self.transform = nn.ModuleList()
        self.out_dim = out_dim
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=(3,), padding=1)
        for i in range(num_block):
            self.transform.append(ResnetBlockConv1d(hidden_dim, kernel_size=(3,)))
        self.last_conv = nn.Conv1d(hidden_dim, 10, kernel_size=(3,), padding=1)
        self.fc1 = nn.Linear(270, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, out_dim)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc_bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        # x [batch, D, 27]
        bsize = x.shape[0]
        x = self.conv(x)
        for i in range(self.num_block):
            x = self.transform[i](x)
        x = self.last_conv(x)
        x = x.view(bsize, -1)
        if bsize != 1:
            x = F.relu(self.fc_bn1(self.fc1(x)))
            x = F.relu(self.fc_bn2(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        if self.out_dim > 1:
            x = torch.softmax(self.fc3(x), dim=1)
        else:
            x = torch.squeeze(self.fc3(x))
        return x
