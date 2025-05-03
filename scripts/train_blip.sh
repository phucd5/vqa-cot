#!/bin/bash
#SBATCH --job-name=train_blip_trial
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80g
#SBATCH --cpus-per-task=6
#SBATCH --mem=175G
#SBATCH --time=10:00:00
#SBATCH --output=logs/train_blip_trial_%j.txt

# ran in the root dir
pip install -e .
python train.py --image_dir data/images/images --train_data_dir data/gqa_flat_train.json --val_data_dir data/gqa_flat_val.json --output_dir saved_models/blip-gqa-ft --model_type blip