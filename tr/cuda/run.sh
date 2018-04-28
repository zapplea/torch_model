#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="emnlp_baseline"
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --mem=5GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --qos=express

echo "loading"
module load python/3.6.1
module load cudnn/v7.1.2-cuda91
module load cuda/9.1.85
module load pytorch/0.3.1-py36-cuda91
echo "loaded"
python cuda4.py