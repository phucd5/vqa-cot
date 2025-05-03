#!/bin/bash
#SBATCH --job-name=eval_blip
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=logs/eval_blip_trial_%j.txt

# ran in the root dir (vqa-cot)
pip install -e .
python eval.py --model_type blip --model_dir Salesforce/blip2-opt-2.7b --eval_dataset_path data/one_gqa_flat_test.json --prompting_mode direct --output_dir results/direct.json --image_dir data/images/images