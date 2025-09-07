#!/usr/bin/env bash
#SBATCH --job-name=rbc_bellman
#SBATCH --output=slurm-%j.out
#SBATCH --time=00:05:00
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G

set -euo pipefail
echo "[$(date)] Starting job on $HOSTNAME"
make -C "$(dirname "$0")" || true
./bellman_demo
./bellman_fortran || true
echo "[$(date)] Job complete"
