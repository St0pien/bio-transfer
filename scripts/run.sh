#!/bin/bash
#SBATCH --job-name=xgb_search
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm-%j.out

module load GCCcore/11.3.0 Python/3.10.4 CUDA/11.7.0
source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# python3 scripts/train_all_downstream.py \
#     --task "regression" \
#     --gnn /net/tscratch/people/plgvltkv/bio-transfer/data/gnn.pt

# python3 scripts/train_all_downstream.py \
#     --task "classification" \
#     --threshold 7.5 \
#     --gnn /net/tscratch/people/plgvltkv/bio-transfer/data/gnn.pt

# python3 scripts/train_all_downstream.py \
#     --task "regression" \
#     --gnn /net/tscratch/people/plgvltkv/bio-transfer/data/filtered_gnn.pt


python3 scripts/train_all_downstream.py \
    --task "classification" \
    --threshold 7.5 \
    --gnn /net/tscratch/people/plgvltkv/bio-transfer/data/filtered_gnn.pt
