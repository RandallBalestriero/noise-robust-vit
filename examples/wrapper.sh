#!/bin/bash
# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES
eval "$(/private/home/rbalestriero/anaconda3/bin/conda shell.bash hook)"
conda activate torch

python test.py --model $model --dataset $dataset --factors $factors --multiplier $multiplier --lr $lr --run $SLURM_ARRAY_TASK_ID --augmentation $augmentation --optimizer $optimizer
