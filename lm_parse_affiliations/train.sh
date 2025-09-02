#!/bin/sh
#SBATCH --job-name=grpo-affiliation-parsing
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH -A marlowe-m000152-pm03
#SBATCH --gpus=7
#SBATCH --exclusive
#SBATCH --time=24:00:00

module load conda
. "$(conda info --base)/etc/profile.d/conda.sh"
conda activate comet
cd /scratch/m000152/comet/reconcile-pkp-beacon-journals-w-openalex-affiliation-metadata/lm_parse_affiliations

accelerate launch --num_processes=7 train.py