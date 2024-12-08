#!/bin/bash --login

## The #SBATCH lines are read by SLURM for options.
## In the lines below we ask for a single node, one task for that node, and one cpu for each task.
#SBATCH --nodes=2
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24

#SBATCH --partition=cs
## Time is the estimated time to complete, in this case 5 hours.
## SBATCH --time=7-00:00:00

## We expect no more than 2GB of memory to be needed
#SBATCH --mem=100GB

## Request GPU card
##SBATCH --gres=gpu:rtx8000:2

## To make them easier to track, it's best to name jobs something recognizable.
## You can then use the name to look up reports with tools like squeue.
#SBATCH --job-name=CLS_Embedding

## These lines manage mail alerts for when the job ends and who the email should be sent to.
#SBATCH --mail-type=END
#SBATCH --mail-user=jy4057@stern.nyu.edu

## This places the standard output and standard error into the same file, in this case slurm_<job_id>.out
#SBATCH --output=slurm_%j.out
## SBATCH --error=slurm_%j_error.out
## SBATCH --open-mode=append

## First we ensure a clean environment by purging the current one
module purge
module load anaconda3/2020.07

## Activate virtual enviornment
conda activate ./penv

## export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python CLS_Embedding.py