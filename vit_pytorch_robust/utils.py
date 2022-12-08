import torch as ch
import random
import os
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    ch.manual_seed(seed)
    ch.cuda.manual_seed(seed)
    ch.cuda.manual_seed_all(seed)
    ch.backends.cudnn.deterministic = True


class SinkhornAttention(ch.jit.ScriptModule):
    def __init__(self, dim: int = -1, sinkhorn_iterations: int = 3) -> None:
        super().__init__()
        self.dim = dim
        self.sinkhorn_iterations = sinkhorn_iterations

    def forward(self, Q: ch.Tensor) -> ch.Tensor:
        Q = ch.softmax(Q, dim=self.dim)
        for _ in range(self.sinkhorn_iterations):
            Q = Q.div(ch.sum(Q, dim=-1, keepdim=True))
            Q = Q.div(ch.sum(Q, dim=-2, keepdim=True))
        return Q


if __name__ == "__main__":
    a = ch.rand(4, 4)
    print(a)
    module = SinkhornAttention()
    ds_a = module(a)
    print(ds_a.sum(0), ds_a.sum(1))
