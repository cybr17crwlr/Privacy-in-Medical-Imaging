#!/usr/bin/env zsh
#SBATCH --job-name=AUG256BS
#SBATCH --partition=instruction
#SBATCH --requeue  
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1-12:00:00
#SBATCH --output="Augment/results_base/EFFNET_256-%j.txt"
#SBATCH -G 2

cd ~/

cd $SLURM_SUBMIT_DIR

module load anaconda/full
bootstrap_conda
conda activate basestats

python Augment/EFFNET_256_BS.py
