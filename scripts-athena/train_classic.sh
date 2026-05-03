#!/bin/bash
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00 
#SBATCH --output=logs/slurm-%j.out

module load Python/3.10.4 CUDA/11.7.0

cd $SCRATCH/bio-transfer

source .venv/bin/activate

pip install xgboost
python $SCRATCH/bio-transfer/train_classic.py