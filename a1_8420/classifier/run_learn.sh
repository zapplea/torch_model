#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="a1_8420"
#SBATCH --time=5:59:00
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --qos=express

echo "loading"
module load python/3.6.1
module load cudnn/v7.1.2-cuda91
module load cuda/9.1.85
module load pytorch/0.3.1-py36-cuda91
echo "loaded"

if test $1 = "cs";
then
    echo rm report.cs
    rm ../report/report_cascade.txt
    python cascading_learn.py
elif test $1 = "pt";
then
    echo rm report.pt
    rm ../report/report_proto_with_share.txt
    rm ../report/report_proto.txt
    python prototypical_learn.py
fi