#!/bin/bash
# submit.sh
# 用法: sbatch submit.sh configs/my_experiment.toml

#SBATCH --job-name CPC-test
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 6
#SBATCH --partition debug
#SBATCH --output logs/%j.out
#SBATCH --error logs/%j.err

CONFIG_FILE=$1

julia --project=. scripts/run-CPCMEM.jl --config configs//test_run.toml