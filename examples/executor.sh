#!/bin/bash

# LOAD YOUR CONDA/VIRTUAL ENV FIRST
module unload anaconda3
module load anaconda3/2020.11-nold
source activate /private/home/rbalestriero/.conda/envs/ffcv
nvidia-smi
s=$((($RANDOM % 100)/50))
sleep $s
python CIFAR100.py --checkpoint-dir $dir -wd $wd -bs $bs -lr $lr --heads $heads --depth $depth --dim $dim --mlp_dim $mlp_dim --epochs 100 -ps $ps