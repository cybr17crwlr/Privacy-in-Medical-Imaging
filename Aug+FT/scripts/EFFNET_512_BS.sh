#!/usr/bin/env zsh
#SBATCH --job-name=basestats_efficientnet_with_augmentation
#SBATCH --partition=instruction
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0-07:00:00
#SBATCH --output="results/BS_EFFNET_AUG_512_FT-%j.txt"
#SBATCH -G 1

cd ~/

cd $SLURM_SUBMIT_DIR

module load anaconda/full
bootstrap_conda
conda activate basestats

python BS_EFFNET_AUG_512_FT.py
