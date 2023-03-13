#!/bin/bash

gpus=1
epochs=200
name=CIFAR10
train_path="/private/home/rbalestriero/DATASETS/${name}/train_raw.ffcv"
val_path="/private/home/rbalestriero/DATASETS/${name}/val_raw.ffcv"
folder="/checkpoint/rbalestriero/VC_ssl_sup/${name}"
lr="0.001"
wd="0.001"
arch=resnet18
for temperature in 0.01 0.1 1 10 100
do
    python sup_ssl.py  --slurm-partition devlab --process-name PROJ --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size 1024 --learning-rate $lr --weight-decay $wd --strength 1 --train-dataset-path $train_path --val-dataset-path $val_path --eval-each-epoch
done
