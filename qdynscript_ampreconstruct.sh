#!/bin/bash
#SBATCH -J ampreconstruct
#SBATCH -n 2
#SBATCH -N 1
#SBATCH -t 0-05:00
#SBATCH -p shared
#SBATCH --mem=42000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kis@mit.edu 
#SBATCH --open-mode=append
#SBATCH -o /n/scratchlfs/demler_lab/kis/genPol_data/std/amprecon%A_%a.out
#SBATCH -e /n/scratchlfs/demler_lab/kis/genPol_data/std/amprecon%A_%a.err

module load Anaconda3/5.0.1-fasrc01
source activate anaclone
python ampreconstruct_cluster.py