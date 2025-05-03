import json
import logging
import glob
import argparse
import os
import random
from collections import defaultdict

from utils.utils import print_json, setup_logging

import torch
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple


logger = logging.getLogger("data_prepoc")


def flatten_gqa_data(
    data_path: str, k: int = -1, seed: int = 42
) -> List[Dict[str, str]]:
    """
    Load GQA and return a list of dicts with keys: question_id, question, answer, image_path, full_answer

    If k>0: perform a stratified sample (by `local` group) of size k

    Args:
        data_path (str): path to the GQA data file
        k (int): number of samples to select, -1 means full dataset
    """
    random.seed(seed)
    gqa = list(json.load(open(data_path)).items())
    n = len(gqa)

    logger.info(f"Loading GQA data from {data_path} with len: {len(gqa)}, k: {k}")
    if k <= 0 or k >= n:
        selected = gqa
    else:
        # bucket by local group
        buckets = defaultdict(list)
        for qid, info in gqa:
            buckets[info["groups"]["local"]].append((qid, info))

        # sample
        sampled = []
        for grp, bucket in buckets.items():
            cnt = max(1, int(len(bucket) * k / n))
            cnt = min(cnt, len(bucket))
            sampled.extend(random.sample(bucket, cnt))

        # ensure k
        if len(sampled) < k:
            seen = {qid for qid, _ in sampled}
            remaining = [pair for pair in gqa if pair[0] not in seen]
            sampled.extend(random.sample(remaining, k - len(sampled)))
        elif len(sampled) > k:
            sampled = random.sample(sampled, k)

        selected = sampled

    logger.info(f"Finished processing GQA data with len f{len(selected)}")
    return [
        {
            "question_id": qid,
            "question": info["question"],
            "answer": info["answer"],
            "image_path": info["imageId"] + ".jpg",
            "full_answer": info["fullAnswer"],
        }
        for qid, info in tqdm(selected, desc="Processing GQA")
    ]


def generate_label_mapping(datasets: List) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Generate label mapping for the GQA data given a list of datasets

    Args:
        dataset list(Dataset): Hugging Face Dataset object containing question data
                           (question_id, question, answer, image_path)
    """
    unique_answer = set()
    for dataset in datasets:
        unique_answer.update(set(dataset["answer"]))
    label2id = {label: i for i, label in enumerate(sorted(unique_answer))}
    id2label = {i: label for i, label in enumerate(sorted(unique_answer))}
    return label2id, id2label


def preprocess_data_classification(
    examples: Dict[str, List],
    processor,
    label2id: Dict[str, int],
    image_dir: str,
):
    """
    Preprocess the data for classification task by passing Q/A to the VILT processor
    and creating one-hot encoded labels from the answer string

    Args:
        examples (Dict[str, List]): dict containing lists of ('question_id', 'question', 'answer', 'image_path', "full_answer")
        processor (ViltProcessor): VILT processor to process the image and text inputs
        label2id (Dict[str, int]): Mapping from label strings to their corresponding IDs
        image_dir (str): Base directory for the images

    Returns:
        Dict[str, torch.Tensor]: Processed data including input_ids, attention_mask, pixel_values, and labels
    """
    images = [
        Image.open(os.path.join(image_dir, image_path)).convert("RGB")
        for image_path in examples["image_path"]
    ]
    questions = examples["question"]

    logger.info(f"Processing {len(questions)} questions and {len(images)} images")
    encoding = processor(
        images=images,
        text=questions,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    num_labels = len(label2id)
    batch_size = len(examples["answer"])
    one_hot_labels = torch.zeros(batch_size, num_labels)

    # one hot encoding
    for i, answer in enumerate(examples["answer"]):
        one_hot_labels[i, label2id[answer]] = 1.0

    encoding["labels"] = one_hot_labels
    return encoding


def preprocess_data_generation(examples: Dict[str, List], processor, image_dir: str):
    """
    Load the data with the model for fine-tuning
    Args:
        train_data_path (str): train data path for json of data containing (question_id, question, answer, image_path, full_answer)
        val_data_path (str): train data path for json of data containing (question_id, question, answer, image_path, full_answer)
    """
    # load images & build prompts
    images = [
        Image.open(os.path.join(image_dir, p)).convert("RGB")
        for p in examples["image_path"]
    ]
    prompts = [f"Question: {q} Answer:" for q in examples["question"]]
    answers = examples["full_answer"]

    max_src_length = min(processor.tokenizer.model_max_length, 384)

    model_inputs = processor(
        images=images,
        text=prompts,
        padding=False,
        truncation=True,
        max_length=max_src_length,
    )

    with processor.tokenizer.as_target_tokenizer():
        target_enc = processor.tokenizer(
            answers,
            padding=False,
            truncation=True,
            max_length=max_src_length,
        )

    # mask pad‐tokens → -100 in each label sequence
    labels = []
    for seq in target_enc["input_ids"]:
        labels.append([
            token_id if token_id != processor.tokenizer.pad_token_id else -100
            for token_id in seq
        ])

    model_inputs["labels"] = labels
    return model_inputs

# helper for main script
def preprocess_question_data(input_files: List[str], output_dir: str, k: int = -1):
    """
    Loop over the data files and flatten the data

    Args:
        input_files (List[str]): List of input file paths
        output_dir (str): Output directory for the flattened data
        k: flatten the data to k length, -1 means full dataset
    """
    all_gqa_data = []

    for file_path in input_files:
        logging.info(f"Processing {file_path}")
        gqa_data = flatten_gqa_data(file_path, k=k)
        all_gqa_data.extend(gqa_data)

    with open(output_dir, "w") as f:
        json.dump(all_gqa_data, f)

    logging.info(f"All question data saved to {output_dir}")
    print_json(output_dir, k=5)


if __name__ == "__main__":
    setup_logging("logs/data_preproc.log")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_path",
        type=str,
        help="Path to train data, should be a glob pattern",
        required=True,
    )
    parser.add_argument(
        "--test_data_path", type=str, help="Path to test data", required=True
    )
    parser.add_argument(
        "--val_data_path", type=str, help="Path to val data", required=True
    )
    parser.add_argument(
        "--output_train_dir", type=str, default="data/gqa_flat_train.json"
    )
    parser.add_argument(
        "--output_test_dir", type=str, default="data/gqa_flat_test.json"
    )
    parser.add_argument("--output_val_dir", type=str, default="data/gqa_flat_val.json")
    parser.add_argument(
        "--k_test",
        type=int,
        default=-1,
        help="Limit test dataset to k elements, will not limit if not set",
    )
    parser.add_argument(
        "--k_train",
        type=int,
        default=-1,
        help="Limit train dataset to k elements, will not limit if not set",
    )
    parser.add_argument(
        "--k_val",
        type=int,
        default=-1,
        help="Limit train dataset to k elements, will not limit if not set",
    )

    parser.add_argument(
        "--process_train", action="store_true", help="Process train set"
    )
    parser.add_argument("--process_test", action="store_true", help="Process test set")
    parser.add_argument("--process_val", action="store_true", help="Process test set")
    args = parser.parse_args()

    train_files = sorted(glob.glob(args.train_data_path))
    test_files = sorted(glob.glob(args.test_data_path))
    val_files = sorted(glob.glob(args.val_data_path))

    if args.process_train:
        preprocess_question_data(
            train_files, output_dir=args.output_train_dir, k=args.k_train
        )

    if args.process_test:
        preprocess_question_data(
            test_files, output_dir=args.output_test_dir, k=args.k_test
        )

    if args.process_val:
        preprocess_question_data(
            val_files, output_dir=args.output_val_dir, k=args.k_val
        )
