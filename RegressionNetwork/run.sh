#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 24:00:00
#SBATCH -o /dev/null
#SBATCH --gres gpu:1

# call your program here
python3 train.py