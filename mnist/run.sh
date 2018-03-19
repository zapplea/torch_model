#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="che313"
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=200GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

echo "loading"
module load python/3.6.1
module load cudnn/v6
module load cuda/8.0.61
module load pytorch/0.3.1-py36
echo "loaded"
python mnist.py