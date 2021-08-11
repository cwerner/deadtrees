#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --partition=ccgp
#SBATCH --time=4:00:0
#SBATCH --output=slurm-%j.log
#SBATCH --chdir=/home/werner-ch/repos/deadtrees/
date;hostname;id;pwd

echo 'activating virtual environment'
micromamba activate deadtrees
which python

config_yaml='/home/werner-ch/repos/deadtrees/sweep.yaml'
echo 'config:' $config_yaml

train_file='/home/werner-ch/repos/deadtrees/scripts/train.py'
echo 'train_file:' $train_file

project_name='deadtrees'
echo 'project_name:' $project_name

echo 'running script'
python sweep_slurm.py $config_yaml $train_file $project_name
