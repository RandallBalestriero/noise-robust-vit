import math
from torch.optim.lr_scheduler import (
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
    ConstantLR,
)
from torch.optim import SGD
import matplotlib.pyplot as plt
import torch


def adjust_learning_rate1(epochs, n_batches, lr):
    max_steps = epochs * n_batches
    warmup_steps = 10 * n_batches
    lrs = []
    print()
    for step in range(max_steps):
        if step < warmup_steps:
            lrs.append(lr * step / warmup_steps)
        else:
            r = (step - warmup_steps) / (max_steps - warmup_steps)
            q = 0.5 * (1 + math.cos(math.pi * r))
            end_lr = lr * 0.001
            lrs.append(lr * q + end_lr * (1 - q))
    return lrs


def adjust_learning_rate2(epochs, n_batches, lr):
    stop = min(10, int(epochs * 0.1))
    T1 = stop * n_batches
    T2 = (epochs - stop) * n_batches
    optimizer = SGD(params=[torch.nn.Parameter(torch.zeros(10))], lr=lr)
    scheduler = SequentialLR(
        optimizer,
        [
            LinearLR(optimizer, 1e-5, 1, total_iters=T1),
            CosineAnnealingLR(optimizer, T_max=T2, eta_min=lr * 1e-3),
        ],
        milestones=[T1],
    )
    lrs = []
    for step in range(n_batches * epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    return lrs


def adjust_learning_rate3(epochs, n_batches, lr):
    optimizer = SGD(params=[torch.nn.Parameter(torch.zeros(10))], lr=lr)
    T = 5 * n_batches
    step1 = LinearLR(optimizer, 1e-3, 1, total_iters=T)
    step2 = ConstantLR(optimizer, factor=1.0, total_iters=2 * T)
    step3 = CosineAnnealingLR(
        optimizer,
        T_max=epochs * n_batches - 3 * T,
        eta_min=lr * 0.005,
    )
    scheduler = SequentialLR(
        optimizer,
        [step1, step2, step3],
        milestones=[T, 3 * T],
    )
    lrs = []
    for step in range(n_batches * epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    return lrs


plt.loglog(adjust_learning_rate1(100, 128, 0.1))
plt.loglog(adjust_learning_rate3(100, 128, 0.1))
plt.savefig("lrs.png")
plt.close()
