import torch as ch
import random
import os
import numpy as np
import signal
import subprocess
from contextlib import closing
import socket
from time import time


def save_all(epoch, model, optimizer, scheduler, path):

    state = dict(
        epoch=epoch + 1,
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
    )
    ch.save(state, path)


def load_all(model, optimizer, scheduler, path):
    ckpt = ch.load(path, map_location="cpu")
    start_epoch = ckpt["epoch"]
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return start_epoch


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_open_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def main(args, main_worker):
    args.ngpus_per_node = ch.cuda.device_count()
    if "SLURM_JOB_ID" in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = "scontrol show hostnames " + os.getenv("SLURM_JOB_NODELIST")
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv("SLURM_NODEID")) * args.ngpus_per_node
        args.world_size = int(os.getenv("SLURM_NNODES")) * args.ngpus_per_node
        args.dist_url = f"tcp://{host_name}:{args.port}"
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = f"tcp://localhost:{args.port}"
        args.world_size = args.ngpus_per_node
    if args.world_size > 1:
        ctx = ch.multiprocessing.spawn(
            main_worker, args=(args,), nprocs=args.ngpus_per_node, join=False
        )
        while not ctx.join():
            time.sleep(4)
    else:
        main_worker(0, args)


def seed_everything(seed, fast=True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    ch.manual_seed(seed)
    ch.cuda.manual_seed(seed)
    ch.cuda.manual_seed_all(seed)
    if fast:
        ch.backends.cudnn.deterministic = False
        ch.backends.cudnn.benchmark = True
    else:
        ch.backends.cudnn.deterministic = True
        ch.backends.cudnn.benchmark = False


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
