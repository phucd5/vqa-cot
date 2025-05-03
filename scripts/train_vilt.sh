#!/bin/bash
#SBATCH --job-name=train_vilt_trial
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=175G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_vilt_trial_%j.txt

# ran in the root dir
pip install -e .
python train.py --image_dir data/images/images --train_data_dir data/gqa_flat_train.json --val_data_dir data/gqa_flat_val.json  --output_dir saved_models/vilt-gqa-ft --model_type vilt