#!/bin/bash
# One job of a campaign on a SLURM cluster. Every job has the same list of units and its own shard.
#
#   csfrbench plan --setting primary --workloads depth4_tree --out units.jsonl
#   sbatch --array=0-15 examples/slurm_job.sbatch units.jsonl runs/depth4_tree
#
# A job that the cluster stops can be started again: the units on record are not run again.
#SBATCH --job-name=csfrbench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1       # one thread per training run
UNITS=$1
OUT=$2
SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
csfrbench run --units "$UNITS" --out "$OUT" --workers "${SLURM_CPUS_PER_TASK:-1}" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}/${SHARDS}"
