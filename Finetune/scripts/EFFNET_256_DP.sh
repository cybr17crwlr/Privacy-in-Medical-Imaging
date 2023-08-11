#!/usr/bin/env zsh
#SBATCH --job-name=FT256DP
#SBATCH --partition=research
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=4-00:00:00
#SBATCH --output="Finetune/results_dp/EFFNET_256-%j.txt"
#SBATCH -G 2

cd ~/

cd $SLURM_SUBMIT_DIR

module load anaconda/full
bootstrap_conda
conda activate privacy

python Finetune/EFFNET_256_DP.py
