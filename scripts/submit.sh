#!/bin/bash
# submit.sh
# 用法: sbatch submit.sh configs/my_experiment.toml

#SBATCH --job-name CPCMEM
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 12
#SBATCH --partition i64m512r
#SBATCH --output logs/%j.out
#SBATCH --error logs/%j.err

CONFIG_FILE=$1

julia --project=. scripts/run-CPCMEM.jl --config $CONFIG_FILE