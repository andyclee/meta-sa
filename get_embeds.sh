#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=eng-instruction
#SBATCH --mail-user=andy2@illinois.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --time=1:00:00

python3 embeddings.py
