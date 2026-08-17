#!/bin/bash

#SBATCH --job-name=laughlin_edge

#SBATCH --partition=cip,inter

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --gres=gpu:a40:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=8G

#SBATCH --time=2-00:00:00

#SBATCH --output=./logs/job_%j.out
#SBATCH --error=./logs/job_%j.err

#SBATCH --mail-type=END
#SBATCH --mail-user=F.Debortoli@lmu.de


srun LaughlinEdge
