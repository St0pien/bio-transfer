#!/bin/bash
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=00:30:00 
#SBATCH --output=logs/slurm-%j.out

module purge
module load Python/3.10.4 CUDA/11.7.0

module load GCCcore/11.3.0 Python/3.10.4 CUDA/11.7.0

cd $SCRATCH/unbranding

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

pip install --upgrade pip
pip install "numpy<2"
pip install torch --index-url https://download.pytorch.org/whl/cu117
pip install pandas
pip install torch_geometric \
  -f https://data.pyg.org/whl/torch-2.0.1+cu117.html

pip install tgdm
pip install rdkit
pip install matplotlib

pip install joblib
pip install scikit-learn

pip install xgboost