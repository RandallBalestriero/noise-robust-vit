#!/bin/bash

GPU=2

bs=256
depth=12
heads=16
ps=32

for wd in 0.003 0.01 0.03
do
    for lr in 0.00003 0.0001 0.0003
    do
        for mlpdim in 512 1024 2048
        do
            for dim in 512 1024 2048
            do
                dir="/checkpoint/rbalestriero/ViT/lastone_lr${lr}_wd${wd}_dim${dim}_mlp${mlpdim}_ps${ps}/"
                config="wd=$wd,bs=$bs,lr=$lr,depth=$depth,heads=$heads,dim=$dim,mlp_dim=$mlpdim,dir=$dir,ps=$ps"
                sbatch --export=$config --job-name=TEST --partition=learnlab --nodes=1 --time=440 --cpus-per-task=10 --ntasks-per-node=$GPU --gpus-per-task=1 --wrap="bash executor.sh"
            done
        done
    done
done
# dir="/checkpoint/rbalestriero/ViT/levit_lr${lr}_wd${wd}_dim${dim}_mlp${mlpdim}/"
# config="wd=$wd,bs=$bs,lr=$lr,depth=$depth,heads=$heads,dim=$dim,mlp_dim=$mlpdim,dir=$dir"
# sbatch --export=$config --job-name=TEST --partition=devlab --nodes=1 --time=240 --cpus-per-task=10 --ntasks-per-node=$GPU --gpus-per-task=1 --wrap="bash executor.sh"


# for dim in 256 512 1024
# do
#     for mlpdim in 512 1024 2048
#     do
#         dir="/checkpoint/rbalestriero/ViT/simplevit_lr${lr}_wd${wd}_dim${dim}_mlp${mlpdim}/"
#         config="wd=$wd,bs=$bs,lr=$lr,depth=$depth,heads=$heads,dim=$dim,mlp_dim=$mlpdim,dir=$dir"
#         sbatch --export=$config --job-name=TEST --partition=devlab --nodes=1 --time=240 --cpus-per-task=10 --ntasks-per-node=$GPU --gpus-per-task=1 --wrap="bash executor.sh"
#     done
# done