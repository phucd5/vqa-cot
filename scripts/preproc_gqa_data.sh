#!/bin/bash
#SBATCH --job-name=prepoc_gqa_data
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/vqa_preprocess_data_%j.out

# ran in the root dir (vqa-cot)
pip install -e .
python data/data_preprocess.py \
  --train_data_path "data/questions/train_balanced_questions.json" \
  --test_data_path "data/questions/testdev_balanced_questions.json" \
  --val_data_path "data/questions/val_balanced_questions.json" \
  --k_train 40000 \
  --k_test 5000 \
  --k_val 5000 \
  --process_train \
  --process_test \
  --process_val
