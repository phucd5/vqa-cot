#!/bin/bash
#SBATCH --job-name=eval_vilt
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_vilt_trial_%j.txt

# ran in the root dir (vqa-cot)
pip install -e .
python eval.py --model_type vilt --model_dir saved_models/vilt-gqa-ft --eval_dataset_path data/one_gqa_flat_test.json --prompting_mode cot --output_dir results/vilt_direct.json --image_dir data/images/images --openai_model_name gpt-4.1-2025-04-14