#!/usr/bin/env zsh
#SBATCH --job-name=FT256BS
#SBATCH --partition=instruction
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=1-12:00:00
#SBATCH --output="Finetune/results_base/EFFNET_256-%j.txt"
#SBATCH -G 2

cd ~/

cd $SLURM_SUBMIT_DIR

module load anaconda/full
bootstrap_conda
conda activate basestats

python FineTune/EFFNET_256_BS.py
