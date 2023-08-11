#!/usr/bin/env zsh
#SBATCH --job-name=DPMIA
#SBATCH --partition=instruction
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=6-00:00:00
#SBATCH --output="Augment/results_dp/EFFNET_256_MIA-%j.txt"
#SBATCH -G 1

cd ~/

cd $SLURM_SUBMIT_DIR

module load anaconda/full
bootstrap_conda
conda activate privacy

python Augment/EFFNET_256_DP_MIA.py
