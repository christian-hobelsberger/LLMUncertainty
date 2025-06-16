#!/bin/bash
#SBATCH -J my_job              # Job name
#SBATCH -o my_job.out          # Standard output file
#SBATCH -e my_job.err          # Standard error file
#SBATCH -p lrz-hgx-h100-94x4   # Partition name
#SBATCH --gres=gpu:1           # Request 1 GPU
#SBATCH --time=0-10:00:00      # Maximum runtime (D-HH:MM:SS)
#SBATCH --cpus-per-task=4      # Number of CPU cores per task
#SBATCH --mem=32G              # Memory to allocate

# Activate virtual environment (if needed)
# source ~/myenv/bin/activate

# Run your program
python llm_model_wrappers.py
