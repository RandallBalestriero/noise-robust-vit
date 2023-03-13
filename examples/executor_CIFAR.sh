#!/bin/bash

train_path="/private/home/rbalestriero/DATASETS/CIFAR100/train_raw.ffcv"
val_path="/private/home/rbalestriero/DATASETS/CIFAR100/val_raw.ffcv"
train_path="/private/home/rbalestriero/DATASETS/CIFAR10/train_raw.ffcv"
val_path="/private/home/rbalestriero/DATASETS/CIFAR10/val_raw.ffcv"
gpus=1
epochs=200
strength=3
arch=resnet18
lr=0.001
folder="/checkpoint/rbalestriero/REVOLUTION3/SMARTINIT_CIFAR10"
for wd in 0.01 0.05
do
    # python simpler_randomlabel.py --label-smoothing 0.8 --process-name INIT$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size 256 --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --smart-init
    python simpler_randomlabel.py --label-smoothing 0.1 --process-name INIT$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size 256 --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --smart-init
    python simpler_randomlabel.py --label-smoothing 0.95 --process-name INIT$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size 256 --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path --smart-init
    python simpler_randomlabel.py --label-smoothing 0.8 --process-name INIT$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size 256 --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path
    python simpler_randomlabel.py --label-smoothing 0.1 --process-name INIT$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size 256 --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path
    python simpler_randomlabel.py --label-smoothing 0.95 --process-name INIT$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size 256 --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path
done

# for strength in 3
# do
#     if [[ "$train_path" == "/private/home/rbalestriero/DATASETS/CIFAR10/train_raw.ffcv" ]];then
#         folder="/checkpoint/rbalestriero/REVOLUTION2/ARCH_CIFAR10_${strength}"
#     elif [[ "$train_path" == "/private/home/rbalestriero/DATASETS/CIFAR100/train_raw.ffcv" ]];then
#         folder="/checkpoint/rbalestriero/REVOLUTION2/ARCH_CIFAR100_${strength}"
#     fi
#     for arch in resnet18 resnet34 resnet50 resnet101 resnet152
#     do
#         lr="0.001"
#         wd="0.2"
#         python simpler_randomlabel.py --process-name $arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size 256 --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path
#         lr="0.001"
#         wd="0.05"
#         python simpler_randomlabel.py --slurm-partition devlab --process-name $arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size 256 --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path
#         for lr in "0.00005" "0.0002" "0.001"
#         do
#             for wd in "0.0" "0.001" "0.01"
#             do
#                 python simpler_randomlabel.py --process-name $arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size 256 --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path
#             done
#         done
#     done
# done

if [[ "$train_path" == "/private/home/rbalestriero/DATASETS/CIFAR10/train_raw.ffcv" ]];then
    folder="/checkpoint/rbalestriero/REVOLUTION3/ARCH_CIFAR10_${strength}"
elif [[ "$train_path" == "/private/home/rbalestriero/DATASETS/CIFAR100/train_raw.ffcv" ]];then
    folder="/checkpoint/rbalestriero/REVOLUTION3/ARCH_CIFAR100_${strength}"
fi
# for arch in resnet34 resnet152 # resnet18 resnet34 resnet50 resnet101 resnet152
# do
#     # lr="0.001"
#     # wd="0.2"
#     # python simpler_randomlabel.py --process-name $arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size 256 --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path
#     lr="0.001"
#     wd="0.05"
#     python simpler_randomlabel.py --label-smoothing 0.8 --process-name CTEN$arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size 256 --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path
#     # for lr in "0.00005" "0.0002" "0.001"
#     # do
#     #     for wd in "0.0" "0.001" "0.01"
#     #     do
#     #         python simpler_randomlabel.py --process-name $arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size 256 --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path
#     #     done
#     # done
# done

# epochs=10000
# arch=alexnet
# for lr in "0.00005" "0.00001"
# do
#     for wd in "0.05"
#     do
#         python simpler_randomlabel.py --label-smoothing 0.8 --process-name $arch --gpus-per-node $gpus --folder $folder --add-version --epochs $epochs --float16 --architecture $arch --batch-size 256 --learning-rate $lr --weight-decay $wd --strength $strength --train-dataset-path $train_path --val-dataset-path $val_path
#     done
# done
