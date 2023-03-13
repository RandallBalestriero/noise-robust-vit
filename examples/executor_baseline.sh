#!/bin/bash

# the below one is good and reached 61%
# python noise-robust-vit/examples/CIFAR100.py --folder TODELETE/swin_notrobust_0005 --learning-rate 0.0005

# this one is the best so far but only 45%
# python noise-robust-vit/examples/CIFAR100.py --folder TODELETE/swin_robust_00005 --robust --learning-rate 0.00005



# python noise-robust-vit/examples/CIFAR100.py --folder TODELETE/swin_robust_00008 --robust --learning-rate 0.00008
# python noise-robust-vit/examples/CIFAR100.py --folder TODELETE/swin_robust_0004 --robust --learning-rate 0.0004
# python noise-robust-vit/examples/CIFAR100.py --folder TODELETE/swin_robust_0001 --robust --learning-rate 0.0001

# python noise-robust-vit/examples/CIFAR100.py --folder TODELETE/swin_robust_0005 --robust --learning-rate 0.0005
# python noise-robust-vit/examples/CIFAR100.py --folder TODELETE/swin_notrobust_002 --learning-rate 0.002


# python noise-robust-vit/examples/CIFAR100.py --folder TODELETE/swin_robust_00001 --robust --learning-rate 0.00001
# python noise-robust-vit/examples/CIFAR100.py --folder TODELETE/levit_robust_00001 --architecture levit --robust --learning-rate 0.00001
# python noise-robust-vit/examples/CIFAR100.py --folder TODELETE/levit_notrobust_0005 --architecture levit --learning-rate 0.0005
# python noise-robust-vit/examples/CIFAR100.py --folder TODELETE/levit_notrobust_00001 --architecture levit --learning-rate 0.00001
# python noise-robust-vit/examples/CIFAR100.py --folder TODELETE/levit_robust_002 --architecture levit --robust --learning-rate 0.002
# python noise-robust-vit/examples/CIFAR100.py --folder TODELETE/levit_robust_0005 --architecture levit --robust --learning-rate 0.0005
# for arch in resnet18
# do
#     for noise in 0 0.01 0.1 0.2
#     do
#         for strength in 2
#         do
#             # python nowak.py --folder ../../NOWAK/crossval --add-version --float16 --architecture $arch --noise-std $noise --strength $strength
#             python nowak.py --folder ../../NOWAK/crossval --add-version --float16 --architecture $arch --noise-std $noise --strength $strength --improved
#         done
#     done
# done

# for arch in resnet18
# do
#     for width in 64 128 512 2048
#     do
#         for depth in 2 4 6
#         do
#             for proba in 0.2 0.4 0.6
#             do
#                 python randomlabel.py --folder ../../RANDOM/dropout --sync-batchnorm --add-version --float16 --architecture $arch --projector-depth $depth --projector-width $width --proba $proba
#             done
#         done
#     done
# done

arch="resnet18"

for i in {1..100}
do
    depths[0]="0"
    depths[1]="1"
    depths[2]="2"
    depth=$[ $RANDOM % 3 ]

    batchs[0]="256"
    batchs[1]="512"
    batchs[2]="1024"
    batch=$[ $RANDOM % 3 ]


    lrs[0]="0.0002"
    lrs[1]="0.0005"
    lrs[2]="0.001"
    lrs[3]="0.002"
    lr=$[ $RANDOM % 4 ]

    widths[0]="512"
    widths[1]="2048"
    widths[2]="3096"
    widths[3]="4096"
    widths[4]="8192"
    width=$[ $RANDOM % 5 ]

    probas[0]="0.0"
    probas[1]="0.01"
    probas[2]="0.05"
    proba=$[ $RANDOM % 3 ]

    smoothings[0]="0.0"
    smoothings[1]="0.01"
    smoothings[2]="0.1"
    smoothings[3]="0.2"
    smoothings[4]="0.3"
    smoothing=$[ $RANDOM % 5 ]


    # losss[0]="ce"
    # losss[1]="l1"
    # losss[2]="l2"
    # losss[3]="sce"
    # losss[4]="bce"
    # losss[5]="sboot"
    # loss=$[ $RANDOM % 6 ]
    losss[0]="ce"
    losss[1]="sce"
    losss[2]="sboot"
    loss=$[ $RANDOM % 3 ]

    betas[0]="0.99"
    betas[1]="0.95"
    betas[2]="0.9"
    betas[3]="0.7"
    betas[4]="0.5"
    beta=$[ $RANDOM % 5 ]
    

    optimizers[0]="adam"
    optimizers[1]="adamw"
    optimizer=$[ $RANDOM % 2 ]

    wds[0]="0.0"
    wds[1]="0.001"
    wds[2]="0.01"
    wds[3]="0.05"
    wds[4]="0.1"
    wd=$[ $RANDOM % 5 ]

    echo "width:${widths[$width]}, proba:${probas[$proba]}, loss:${losss[$loss]}, smoothing:${smoothings[$smoothing]}, optimizer:${optimizers[$optimizer]}, decay:${wds[$wd]}"
    python randomlabel.py --folder ../../RANDOM_CORRECTED_DA_V2/search --sync-batchnorm --add-version --epochs 400 --float16 --architecture $arch --projector-depth ${depths[$depth]} --projector-width ${widths[$width]} --proba ${probas[$proba]} --loss ${losss[$loss]} --label-smoothing ${smoothings[$smoothing]} --optimizer ${optimizers[$optimizer]} --weight-decay ${wds[$wd]} --beta ${betas[$beta]} --batch-size ${batchs[$batch]} --learning-rate ${lrs[$lr]} --strength 3
done

# for arch in resnet18 resnet50
# do
#     for strength in 3
#     do
#         python baseline.py --folder ../../BASELINES/test --sync-batchnorm --add-version --float16 --epochs 200 --architecture $arch --weight-decay 0.05 --batch-size 256 --learning-rate 0.0005 --strength $strength --dataset-path "/private/home/rbalestriero/DATASETS/Food101"
#         python baseline.py --folder ../../BASELINES/test --sync-batchnorm --add-version --float16 --epochs 200 --architecture $arch --weight-decay 0.05 --batch-size 256 --learning-rate 0.0005 --strength $strength --dataset-path "/private/home/rbalestriero/DATASETS/Flowers102"
#         python baseline.py --folder ../../BASELINES/test --sync-batchnorm --add-version --float16 --epochs 200 --architecture $arch --weight-decay 0.05 --batch-size 256 --learning-rate 0.0005 --strength $strength --dataset-path "/private/home/rbalestriero/DATASETS/FGVCAircraft"
#         python baseline.py --folder ../../BASELINES/test --sync-batchnorm --add-version --float16 --epochs 200 --architecture $arch --weight-decay 0.05 --batch-size 256 --learning-rate 0.0005 --strength $strength --dataset-path "/private/home/rbalestriero/DATASETS/OxfordIIITPet"
#         python baseline.py --folder ../../BASELINES/test --sync-batchnorm --add-version --float16 --epochs 200 --architecture $arch --weight-decay 0.05 --batch-size 256 --learning-rate 0.0005 --strength $strength --dataset-path "/private/home/rbalestriero/DATASETS/DTD"
#         python baseline.py --folder ../../BASELINES/test --sync-batchnorm --add-version --float16 --epochs 200 --architecture $arch --weight-decay 0.05 --batch-size 256 --learning-rate 0.0005 --strength $strength --dataset-path "/private/home/rbalestriero/DATASETS/Country211"
#     done
# done