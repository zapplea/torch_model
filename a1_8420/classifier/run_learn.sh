#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="emnlp_baseline"
#SBATCH --time=5:59:00
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --qos=express

echo "loading"
module load python/3.6.1
module load cudnn/v7.1.2-cuda91
module load cuda/9.1.85
module load pytorch/0.3.1-py36-cuda91
echo "loaded"

if test $1 = "cs";
then
    echo rm report
    rm /datastore/liu121/torch_data/a1_8420/report
    python cascading_learn.py
elif test $1 = "pt";
then
    rm /datastore/liu121/torch_data/a1_8420/report
    python prototypical_learn.py
fi