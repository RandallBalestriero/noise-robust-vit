#!/bin/bash

gpus=1
strength=3
lr=0.001
wd=0.05
bs=256


train_path="/private/home/rbalestriero/DATASETS/CIFAR100/train_raw.ffcv"
val_path="/private/home/rbalestriero/DATASETS/CIFAR100/val_raw.ffcv"
epochs=4000
for arch in alexnet #resnet18 resnet50 resnet101
do
    for ls in 0.8 #0.0 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99
    do
        folder="/checkpoint/rbalestriero/REVOLUTION2/LS_CV_CIFAR100/ALEXNET"
        python simpler_randomlabel.py --label-smoothing $ls --process-name LS$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path
    done
done

# train_path="/private/home/rbalestriero/DATASETS/TINYIMAGENET/train_raw.ffcv"
# val_path="/private/home/rbalestriero/DATASETS/TINYIMAGENET/val_raw.ffcv"
# epochs=2000
# for arch in convnext_tiny convnext_small convnext_base
# do
#     for ls in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99
#     do
#         folder="/checkpoint/rbalestriero/REVOLUTION2/LS_CV_TINYIMAGENET/"
#         python simpler_randomlabel.py --label-smoothing $ls --process-name LS$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path
#     done
# done