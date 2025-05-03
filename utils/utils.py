import json
import logging
import os

import torch


def setup_logging(log_file="logs/vqa-cot.log", level=logging.INFO):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def get_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def print_json(data_path: str, k: int = 5):
    """
    Print first k items from a JSON file

    Args:
        data_path (str): path to the JSON file
        k (int): umber of items to print (Default: 5)
    """
    with open(data_path, "r") as f:
        data = json.load(f)

    for i, item in enumerate(data):
        if i >= k:
            break
        print(item)
