#!/bin/bash
#SBATCH --job-name=4_workers            # Job name
#SBATCH --output=scripts/logs/output_%j.log          # Output log file (%j will be replaced by job ID)
#SBATCH --error=scripts/logs/error_%j.log            # Error log file
#SBATCH --time=12:00:00                 # Time limit (hh:mm:ss)
#SBATCH --ntasks=1                      # Number of tasks (usually set to 1 for single-node jobs)
#SBATCH --mem=32G                        # Memory per node
#SBATCH --gpus=1                         # Number of GPUs per node

# Run your application
python transformer.py --wandb_project wash_big --max_local_step 10000 --num_workers 4 --learning_rate 0.0005 --cosine_anneal --save_dir outputs/transformer_4