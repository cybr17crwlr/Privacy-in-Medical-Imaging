#!/usr/bin/env zsh
#SBATCH --job-name=AUG256DP
#SBATCH --partition=instruction
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=6-00:00:00
#SBATCH --output="Augment/results_dp/EFFNET_256-%j.txt"
#SBATCH -G 2

cd ~/

cd $SLURM_SUBMIT_DIR

module load anaconda/full
bootstrap_conda
conda activate privacy

python Augment/EFFNET_256_DP.py
