import os
import math
import random
from numbers import Number
import logging

import numpy as np
import torch

# use to approx logdet
from torch.backends import cudnn


def random_choice(action):
    if isinstance(action, bool):
        return action
    if isinstance(action, int):
        tmp = torch.zeros(27)
        tmp[action] = 1
        return tmp
    if isinstance(action, float):
        flag = np.random.choice([0, 1], 1, p=[action, 1-action])
        return True if flag == 0 else False
    if not isinstance(action, torch.Tensor):
        p = torch.tensor(action)
    else:
        p = action.detach().cpu()
    p = p.view(-1)
    if p.shape[-1] == 27:
        p = p.numpy()
        p = p / p.sum()
        play = np.random.choice(np.arange(0, 27), 1, p=p)
        return play
    elif p.shape[-1] == 2:
        p = p.numpy()
        p = p / p.sum()
        flag = np.random.choice([0, 1], 1, p=p)
        return True if flag == 0 else False
    else:
        assert p.shape[-1] == 1
        r = torch.rand(1).to(p)
        return True if p > r else False


def trace_df_dx_hutchinson(f, x, noise, no_autograd):
    """
    Hutchinson's trace estimator for Jacobian df/dx, O(1) call to autograd
    """
    if no_autograd:
        # the following is compatible with checkpointing
        torch.sum(f * noise).backward()
        # torch.autograd.backward(tensors=[f], grad_tensors=[noise])
        jvp = x.grad
        trJ = torch.sum(jvp * noise, dim=[1, 2, 3])
        x.grad = None
    else:
        jvp = torch.autograd.grad(f, x, noise, create_graph=False)[0]
        trJ = torch.sum(jvp * noise, dim=[1, 2, 3])
        # trJ = torch.einsum('bijk,bijk->b', jvp, noise)  # we could test if there's a speed difference in einsum vs sum

    return trJ


def get_mixed_prediction(mixed_prediction, param, mixing_logit, mixing_component=None):
    if mixed_prediction:
        assert mixing_component is not None, 'Provide mixing component when mixed_prediction is enabled.'
        coeff = torch.sigmoid(mixing_logit).to(param)
        param = (1 - coeff) * mixing_component + coeff * param

    return param


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def logabsdet(x):
    """Returns the log absolute determinant of square matrix x."""
    # Note: torch.logdet() only works for positive determinant.
    _, res = torch.slogdet(x)
    return res


def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tensor2numpy(x):
    return x.detach().cpu().numpy()


def orthogonalize_tensor(tensor):
    assert len(tensor.shape) == 2
    # flattened = tensor.new(rows, cols).normal_(0, 1)

    # Compute the qr factorization
    q, r = torch.qr(tensor)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph
    tensor.view_as(q).copy_(q)
    return tensor


def random_orthogonal(size):
    """
    Returns a random orthogonal matrix as a 2-dim tensor of shape [size, size].
    """

    # Use the QR decomposition of a random Gaussian matrix.
    x = torch.randn(size, size)
    q, _ = torch.qr(x)
    return q


def common_init(seed):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + self.val * n
        self.count += n
        self.avg = self.sum / self.count
