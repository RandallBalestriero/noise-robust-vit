#!/bin/bash

train_path="/private/home/rbalestriero/DATASETS/CIFAR100/train_raw.ffcv"
val_path="/private/home/rbalestriero/DATASETS/CIFAR100/val_raw.ffcv"
gpus=1
strength=3
lr=0.001
wd=0.05
bs=256

for arch in resnet101
do
    for epochs in 10000
    do
        # folder="/checkpoint/rbalestriero/REVOLUTION2/EPOCHS_CV_CIFAR100/${arch}/${epochs}"
        # python simpler_randomlabel.py --process-name EPO$arch --label-smoothing 0.1 --gpus-per-node $gpus --folder $folder --sync-batchnorm --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --save-final-model
        # folder="/checkpoint/rbalestriero/REVOLUTION2/EPOCHS_LS_CV_CIFAR100/${arch}/${epochs}"
        # python simpler_randomlabel.py --process-name EPO$arch --label-smoothing 0.4 --gpus-per-node $gpus --folder $folder --sync-batchnorm --add-version --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --save-final-model
        folder="/checkpoint/rbalestriero/REVOLUTION2/EPOCHS_LSLS_CV_CIFAR100/resnet101/10000/62050e1d-e7d7-4293-a708-e2d89986436d/"
        python simpler_randomlabel.py --process-name EPO$arch --label-smoothing 0.8 --gpus-per-node $gpus --folder $folder --sync-batchnorm --epochs $epochs --float16 --architecture $arch --batch-size $bs --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path
    done
done