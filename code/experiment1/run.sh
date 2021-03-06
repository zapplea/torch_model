#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="a1_8420"
#SBATCH --time=5:59:00
#SBATCH --nodes=1
#SBATCH --mem=1GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --qos=express

echo "loading"
module load python/3.6.1
module load cudnn/v7.1.2-cuda91
module load cuda/9.1.85
module load pytorch/0.3.1-py36-cuda91
echo "loaded"

python data_feeder.py