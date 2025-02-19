# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import models as torchvision_models
from torchvision import transforms as pth_transforms
import numpy as np


# adapted from https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf/utils.py#L1


class PCA:
    """
    Class to  compute and apply PCA.
    """

    def __init__(self, dim=256, whit=0.5):
        self.dim = dim
        self.whit = whit
        self.mean = None

    def train_pca(self, cov):
        """
        Takes a covariance matrix (np.ndarray) as input.
        """
        d, v = np.linalg.eigh(cov)
        eps = d.max() * 1e-5
        n_0 = (d < eps).sum()
        if n_0 > 0:
            d[d < eps] = eps

        # total energy
        totenergy = d.sum()

        # sort eigenvectors with eigenvalues order
        idx = np.argsort(d)[::-1][: self.dim]
        d = d[idx]
        v = v[:, idx]

        print("keeping %.2f %% of the energy" % (d.sum() / totenergy * 100.0))

        # for the whitening
        d = np.diag(1.0 / d**self.whit)

        # principal components
        self.dvt = np.dot(d, v.T)

    def apply(self, x):
        # input is from numpy
        if isinstance(x, np.ndarray):
            if self.mean is not None:
                x -= self.mean
            return np.dot(self.dvt, x.T).T

        # input is from torch and is on GPU
        if x.is_cuda:
            if self.mean is not None:
                x -= torch.cuda.FloatTensor(self.mean)
            return torch.mm(
                torch.cuda.FloatTensor(self.dvt), x.transpose(0, 1)
            ).transpose(0, 1)

        # input if from torch, on CPU
        if self.mean is not None:
            x -= torch.FloatTensor(self.mean)
        return torch.mm(torch.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    # launched with submitit on a slurm cluster
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print("Will run the code on one GPU.")
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
    else:
        print("Does not support training without GPU.")
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)


class CopydaysDataset:
    def __init__(self, basedir):
        self.basedir = basedir
        self.block_names = (
            ["original", "strong"]
            + ["jpegqual/%d" % i for i in [3, 5, 8, 10, 15, 20, 30, 50, 75]]
            + ["crops/%d" % i for i in [10, 15, 20, 30, 40, 50, 60, 70, 80]]
        )
        self.nblocks = len(self.block_names)

        self.query_blocks = range(self.nblocks)
        self.q_block_sizes = np.ones(self.nblocks, dtype=int) * 157
        self.q_block_sizes[1] = 229
        # search only among originals
        self.database_blocks = [0]

    def get_block(self, i):
        dirname = self.basedir + "/" + self.block_names[i]
        fnames = [
            dirname + "/" + fname
            for fname in sorted(os.listdir(dirname))
            if fname.endswith(".jpg")
        ]
        return fnames

    def get_block_filenames(self, subdir_name):
        dirname = self.basedir + "/" + subdir_name
        return [
            fname for fname in sorted(os.listdir(dirname)) if fname.endswith(".jpg")
        ]

    def eval_result(self, ids, distances):
        j0 = 0
        for i in range(self.nblocks):
            j1 = j0 + self.q_block_sizes[i]
            block_name = self.block_names[i]
            I = ids[j0:j1]  # block size
            sum_AP = 0
            if block_name != "strong":
                # 1:1 mapping of files to names
                positives_per_query = [[i] for i in range(j1 - j0)]
            else:
                originals = self.get_block_filenames("original")
                strongs = self.get_block_filenames("strong")

                # check if prefixes match
                positives_per_query = [
                    [j for j, bname in enumerate(originals) if bname[:4] == qname[:4]]
                    for qname in strongs
                ]

            for qno, Iline in enumerate(I):
                positives = positives_per_query[qno]
                ranks = []
                for rank, bno in enumerate(Iline):
                    if bno in positives:
                        ranks.append(rank)
                sum_AP += score_ap_from_ranks_1(ranks, len(positives))

            print("eval on %s mAP=%.3f" % (block_name, sum_AP / (j1 - j0)))
            j0 = j1


# from the Holidays evaluation package
def score_ap_from_ranks_1(ranks, nres):
    """Compute the average precision of one search.
    ranks = ordered list of ranks of true positives
    nres  = total number of positives in dataset
    """

    # accumulate trapezoids in PR-plot
    ap = 0.0

    # All have an x-size of:
    recall_step = 1.0 / nres

    for ntp, rank in enumerate(ranks):

        # y-size on left side of trapezoid:
        # ntp = nb of true positives so far
        # rank = nb of retrieved items so far
        if rank == 0:
            precision_0 = 1.0
        else:
            precision_0 = ntp / float(rank)

        # y-size on right side of trapezoid:
        # ntp and rank are increased by one
        precision_1 = (ntp + 1) / float(rank + 1)

        ap += (precision_1 + precision_0) * recall_step / 2.0

    return ap


class ImgListDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, transform=None):
        self.samples = img_list
        self.transform = transform

    def __getitem__(self, i):
        with open(self.samples[i], "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, i

    def __len__(self):
        return len(self.samples)


def is_image_file(s):
    ext = s.split(".")[-1]
    if ext in ["jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp"]:
        return True
    return False


@torch.no_grad()
def extract_features(image_list, model, args):
    transform = pth_transforms.Compose(
        [
            pth_transforms.Resize((args.imsize, args.imsize), interpolation=3),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    tempdataset = ImgListDataset(image_list, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        tempdataset,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        drop_last=False,
        sampler=torch.utils.data.DistributedSampler(tempdataset, shuffle=False),
    )
    features = None
    for samples, index in data_loader:
        samples, index = samples.cuda(non_blocking=True), index.cuda(non_blocking=True)
        feats = model.get_intermediate_layers(samples, n=1)[0].clone()

        cls_output_token = feats[:, 0, :]  #  [CLS] token
        # GeM with exponent 4 for output patch tokens
        b, h, w, d = (
            len(samples),
            int(samples.shape[-2] / model.patch_embed.patch_size),
            int(samples.shape[-1] / model.patch_embed.patch_size),
            feats.shape[-1],
        )
        feats = feats[:, 1:, :].reshape(b, h, w, d)
        feats = feats.clamp(min=1e-6).permute(0, 3, 1, 2)
        feats = (
            nn.functional.avg_pool2d(feats.pow(4), (h, w)).pow(1.0 / 4).reshape(b, -1)
        )
        # concatenate [CLS] token and GeM pooled patch tokens
        feats = torch.cat((cls_output_token, feats), dim=1)

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            features = features.cuda(non_blocking=True)

        # get indexes from all processes
        y_all = torch.empty(
            dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device
        )
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            features.index_copy_(0, index_all, torch.cat(output_l))
    return features  # features is still None for every rank which is not 0 (main)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Copy detection on Copydays")
    parser.add_argument(
        "--data_path",
        default="/path/to/copydays/",
        type=str,
        help="See https://lear.inrialpes.fr/~jegou/data.php#copydays",
    )
    parser.add_argument(
        "--whitening_path",
        default="/path/to/whitening_data/",
        type=str,
        help="""Path to directory with images used for computing the whitening operator.
        In our paper, we use 20k random images from YFCC100M.""",
    )
    parser.add_argument(
        "--distractors_path",
        default="/path/to/distractors/",
        type=str,
        help="Path to directory with distractors images. In our paper, we use 10k random images from YFCC100M.",
    )
    parser.add_argument(
        "--imsize", default=320, type=int, help="Image size (square image)"
    )
    parser.add_argument(
        "--batch_size_per_gpu", default=16, type=int, help="Per-GPU batch-size"
    )
    parser.add_argument(
        "--pretrained_weights",
        default="",
        type=str,
        help="Path to pretrained weights to evaluate.",
    )
    parser.add_argument("--arch", default="vit_base", type=str, help="Architecture")
    parser.add_argument(
        "--patch_size", default=8, type=int, help="Patch resolution of the model."
    )
    parser.add_argument(
        "--checkpoint_key",
        default="teacher",
        type=str,
        help='Key to use in the checkpoint (example: "teacher")',
    )
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""",
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="Please ignore and do not set this argument.",
    )
    args = parser.parse_args()

    init_distributed_mode(args)

    print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )
    cudnn.benchmark = True

    # ============ building network ... ============
    model = torchvision_models.__dict__[args.arch](pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()

    # load pretrained weights
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(
            "Pretrained weights found at {} and loaded with msg: {}".format(
                args.pretrained_weights, msg
            )
        )
    else:
        print("Warning: We use random weights.")

    model.cuda()
    model.eval()

    dataset = CopydaysDataset(args.data_path)

    # ============ Extract features ... ============
    # extract features for queries
    queries = []
    for q in dataset.query_blocks:
        queries.append(extract_features(dataset.get_block(q), model, args))
    if get_rank() == 0:
        queries = torch.cat(queries)
        print(f"Extraction of queries features done. Shape: {queries.shape}")

    # extract features for database
    database = []
    for b in dataset.database_blocks:
        database.append(extract_features(dataset.get_block(b), model, args))

    # extract features for distractors
    if os.path.isdir(args.distractors_path):
        print("Using distractors...")
        list_distractors = [
            os.path.join(args.distractors_path, s)
            for s in os.listdir(args.distractors_path)
            if is_image_file(s)
        ]
        database.append(extract_features(list_distractors, model, args))
    if get_rank() == 0:
        database = torch.cat(database)
        print(
            f"Extraction of database and distractors features done. Shape: {database.shape}"
        )

    # ============ Whitening ... ============
    if os.path.isdir(args.whitening_path):
        print(
            f"Extracting features on images from {args.whitening_path} for learning the whitening operator."
        )
        list_whit = [
            os.path.join(args.whitening_path, s)
            for s in os.listdir(args.whitening_path)
            if is_image_file(s)
        ]
        features_for_whitening = extract_features(list_whit, model, args)
        if get_rank() == 0:
            # center
            mean_feature = torch.mean(features_for_whitening, dim=0)
            database -= mean_feature
            queries -= mean_feature
            pca = PCA(dim=database.shape[-1], whit=0.5)
            # compute covariance
            cov = (
                torch.mm(features_for_whitening.T, features_for_whitening)
                / features_for_whitening.shape[0]
            )
            pca.train_pca(cov.cpu().numpy())
            database = pca.apply(database)
            queries = pca.apply(queries)

    # ============ Copy detection ... ============
    if get_rank() == 0:
        # l2 normalize the features
        database = nn.functional.normalize(database, dim=1, p=2)
        queries = nn.functional.normalize(queries, dim=1, p=2)

        # similarity
        similarity = torch.mm(queries, database.T)
        distances, indices = similarity.topk(20, largest=True, sorted=True)

        # evaluate
        retrieved = dataset.eval_result(indices, distances)
    dist.barrier()
