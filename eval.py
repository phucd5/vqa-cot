import argparse
import collections
import json
import logging
import os
import sys
from typing import Any, Dict, Tuple, List
import uuid
import re

from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from openai import OpenAI

from cot.llm_prompt import (
    get_final_answer,
    get_sub_questions,
    get_sub_questions_sequential,
    extract_blip_answer,
)
from model.base_model import Model
from model.blip_model import BLIPModel
from model.vilt_model import VILTModel
from utils.utils import setup_logging


logger = logging.getLogger("eval_system")


def normalize_text(text: str) -> str:
    text = text.replace('"', "")
    text = re.sub(r"[.,!?;:]", " ", text)  # replace punctuation with space
    return " ".join(
        [word.strip().lower() for word in text.split() if word.strip()]
    )  # join words with space


def write_json(output_dir: str, json_output: str):
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    with open(output_dir, "w") as f:
        json.dump(json_output, f, indent=4)

    logger.info(f"Results written to {output_dir}")


def load_checkpoint(checkpoint_path: str) -> Tuple[List, int, int, set]:
    """
    Load evaluation progress from a checkpoint file

    Args:
        checkpoint_path (str): Path to the checkpoint file

    Returns:
        tuple: (results, correct_count, total_count, processed_samples)
    """
    results = []
    corr, total = 0, 0
    processed_samples = set()

    if not os.path.exists(checkpoint_path):
        logger.info("No checkpoint found. Starting fresh evaluation.")
        return results, corr, total, processed_samples

    try:
        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)

        results = checkpoint_data.get("results", [])
        corr = checkpoint_data.get("correct_count", 0)
        total = checkpoint_data.get("total_count", 0)

        processed_samples = {sample["question"] for sample in results}

        logger.info(
            f"Resuming from checkpoint with {total} samples processed. Current accuracy: {corr/total if total else 0:.4f}"
        )
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}. Starting fresh.")

    return results, corr, total, processed_samples


def save_checkpoint(
    checkpoint_path: str,
    prompting_mode: str,
    openai_model_name: str,
    model_dir: str,
    results: List,
    corr: int,
    total: int,
):
    """
    Save current evaluation progress to a checkpoint file

    Args:
        checkpoint_path (str): path to save the checkpoint
        prompting_mode (str): prompting mode being used
        openai_model_name (str): name of the OpenAI model being used
        model_dir (str): directory of the VQA model
        results (list): list of evaluation results
        corr (int): number of correct predictions
        total (int): total number of predictions
    """
    checkpoint_data = {
        "test_type": f"{prompting_mode}",
        "openai_model_name": openai_model_name,
        "model_dir": model_dir,
        "accuracy": corr / total if total else 0.0,
        "correct_count": corr,
        "total_count": total,
        "results": results,
    }

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f, indent=4)


def setup_checkpoint(output_path: str):
    """
    Setup checkpoint file path based on output JSON file path

    Args:
        output_path (str): Path to the output JSON file

    Returns:
        str or None: Path to checkpoint file, or None if output_path is None
    """
    if not output_path:
        return None

    output_dir = os.path.dirname(output_path)
    filename = os.path.basename(output_path)

    base_name = os.path.splitext(filename)[0]
    checkpoint_name = f"{base_name}_checkpoint.json"

    checkpoint_path = os.path.join(output_dir, checkpoint_name)

    return checkpoint_path


def eval_prompting(
    prompting_mode: str,
    model: Model,
    eval_dataset_path: str,
    image_dir: str,
    client=None,
    openai_model_name: str = "gpt-4.1-2025-04-14",
    output_dir: str = None,
) -> Dict[str, Any]:
    """
    Perform evaluation on the model

    Args:
        prompting_mode (str): prompting mode to use
        model (Model): VQA model
        eval_dataset_path (str): path to the evaluation dataset
        image_dir (str): directory containing the images
        client (OpenAI): OpenAI client
        openai_model_name (str): OpenAI model name
        output_dir (str): output directory for saving results
    Returns:
        dict: evaluation results
    """
    checkpoint_path = setup_checkpoint(output_dir)

    # load checkpoint
    results, corr, total, processed_samples = (
        load_checkpoint(checkpoint_path) if checkpoint_path else ([], 0, 0, set())
    )

    eval_dataset = load_dataset("json", data_files=eval_dataset_path, streaming=True)[
        "train"
    ]

    for qa in eval_dataset:
        try:

            image_path = os.path.join(image_dir, qa["image_path"])
            question, answer = qa["question"], qa["answer"]
            # skip if this sample has already been processed
            if question in processed_samples:
                logger.info(f"Skipping already processed sample: {question}")
                continue

            if prompting_mode == "direct":
                model_prediction, qa_pairs = predict_direct(model, question, image_path)
            elif prompting_mode == "cot":
                model_prediction, qa_pairs = predict_cot(
                    model, question, image_path, client, openai_model_name
                )
            elif prompting_mode == "cot-consistent":
                model_prediction, qa_pairs = predict_cot_consistent(
                    model, question, image_path, client, openai_model_name
                )
            elif prompting_mode == "cot-sequential":
                model_prediction, qa_pairs = predict_cot_sequential(
                    model, question, image_path, client, openai_model_name
                )
            else:
                raise ValueError(
                    f"Invalid prompting mode: {prompting_mode!r}. "
                    f"Please choose one of ['direct', 'cot', 'cot-consistent', 'cot-sequential']"
                )

            if not model_prediction:
                continue

            if isinstance(model, BLIPModel):
                if prompting_mode == "direct":
                    model_prediction = extract_blip_answer(model_prediction)

            answer = answer.lower()
            model_prediction = model_prediction.lower()

            # correctness check
            is_correct = model_prediction == answer

            if not is_correct:
                answer_norm, pred_norm = normalize_text(answer), normalize_text(
                    model_prediction
                )

                if answer_norm == pred_norm:
                    is_correct = True
                else:
                    if len(answer_norm.split()) == 1:
                        is_correct = answer_norm in pred_norm.split()
                    elif len(pred_norm.split()) == 1:
                        is_correct = pred_norm in answer_norm.split()

            if is_correct:
                corr += 1

            sample = {
                "question": question,
                "image_path": image_path,
                "qa_pairs": qa_pairs,
                "expected_answer": answer,
                "model_prediction": model_prediction,
                "is_correct": is_correct,
            }
            results.append(sample)
            processed_samples.add(image_path)
            total += 1

            if checkpoint_path:
                save_checkpoint(
                    checkpoint_path,
                    prompting_mode,
                    openai_model_name,
                    model.model_dir,
                    results,
                    corr,
                    total,
                )

                if total % 10 == 0:
                    logger.info(
                        f"Checkpoint {checkpoint_path} saved after {total} samples. Current accuracy: {corr/total:.4f}"
                    )

        except Exception as e:
            logger.error(f"Error processing sample: {e}")
            exit(1)

    json_output = {
        "test_type": f"{prompting_mode}",
        "openai_model_name": openai_model_name,
        "model_dir": model.model_dir,
        "accuracy": corr / total if total else 0.0,
        "correct_count": corr,
        "total_count": total,
        "results": results,
    }

    return json_output


def predict_direct(
    model: Model, question: str, image_path: str
) -> Tuple[str, Tuple[str, str]]:
    """
    Predicts the answer to a question directly using the VQA model
    Args:
        model (Model): VQA model
        question (str):  complex visual question
        image_path (str): path to the image related to the question
    Returns:
        str: answer, qa pairs (none in this case)
    """
    return model.forward(image_path, question), None


def predict_cot(
    model: Model,
    question: str,
    image_path: str,
    client,
    openai_model_name: str = "gpt-4.1-2025-04-14",
) -> Tuple[str, Tuple[str, str]]:
    """
    Predicts the answer to a question through Chain-of-Thought prompting by generating sub-questions
    Args:
        model (Model): VQA model
        question (str): question to be answered
        image_path (str): path to the image related to the question
        client (OpenAI): OpenAI client for Chain-of-Thought prompting
        openai_model_name (str): OpenAI model name for Chain-of-Thought prompting
    Returns:
        str: answer, qa pairs
    """
    if not client:
        raise ValueError("OpenAI client is required for Chain-of-Thought prompting")

    model_type = "classification" if isinstance(model, VILTModel) else "generation"

    subquestions = get_sub_questions(
        openai_model_name, client, question, model_type=model_type
    )
    qa_pairs = []

    if not subquestions:
        return None, None

    for subquestion in subquestions:
        subanswer = model.forward(image_path, subquestion)
        logger.info(f"Sub-question: {subquestion} \nSub-answer: {subanswer}")
        qa_pairs.append((subquestion, subanswer))

    return get_final_answer(openai_model_name, client, qa_pairs, question), qa_pairs


def predict_cot_consistent(
    model: Model,
    question: str,
    image_path: str,
    client,
    openai_model_name: str = "gpt-4.1-2025-04-14",
    k: int = 5,
) -> Tuple[str, Tuple[str, str]]:
    """
    Predicts the answer to a question through Chain-of-Thought prompting with self-consistency (picking the most common answer)
    Args:
        model (Model): VQA model
        question (str): question to be answered
        image_path (str): path to the image related to the question
        client (OpenAI): OpenAI client
        openai_model_name (str): OpenAI model name
        k (int): the number of reasoning paths to explore. Default is 5
    Returns:
        str: answer, qa pairs
    """
    answers = {}
    qa_pairs = []
    for _ in range(k):
        answer, qa_pair = predict_cot(
            model, question, image_path, client, openai_model_name
        )
        qa_pairs.extend(qa_pair)
        if not answer:
            raise ValueError("Answer failed to generate.")
        if answer not in answers:
            answers[answer] = 1
        else:
            answers[answer] += 1
    if not answers:
        return None, None
    return max(answers, key=answers.get), qa_pairs


def predict_cot_sequential(
    model: Model,
    question: str,
    image_path: str,
    client,
    openai_model_name: str = "gpt-4.1-2025-04-14",
) -> Tuple[str, Tuple[str, str]]:
    """
    Predicts the answer to a visual question through Chain-of-Thought prompting iteratively
    by generating a sub-question getting the answer and then using that answer to generate the next sub-question
    Args:
        model (Model): VQA model
        question (str): question to be answered
        image_path (str): path to the image related to the question
        client (OpenAI): OpenAI client
        openai_model_name (str): OpenAI model name
    Returns:
        str: answer, qa pairs
    """
    if not client:
        raise ValueError("OpenAI client is required for Chain-of-Thought prompting")

    model_type = "classification" if isinstance(model, VILTModel) else "generation"

    subquestions_answer_pair = get_sub_questions_sequential(
        model,
        image_path,
        openai_model_name,
        client,
        question,
        model_type=model_type,
        k=5,
    )
    qa_pairs = []

    if not subquestions_answer_pair:
        return None, None

    qa_pairs = [
        (subquestion, answer) for subquestion, answer in subquestions_answer_pair
    ]

    logger.info(f"Sub-questions and answers: {qa_pairs}")
    return get_final_answer(openai_model_name, client, qa_pairs, question), qa_pairs


if __name__ == "__main__":
    log_file = f"logs/eval_model_{str(uuid.uuid4())[:8]}.log"
    setup_logging(log_file)
    print(f"Log file {log_file}")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompting_mode",
        required=True,
        choices=["direct", "cot", "cot-consistent", "cot-sequential"],
        help="Prompting strategy: direct, cot, cot-consistent, or cot-sequential",
    )
    parser.add_argument(
        "--eval_dataset_path", type=str, help="Path to the evaluation dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../eval_output/eval_results.json",
        help="Output directory",
    )
    parser.add_argument(
        "--openai_model_name",
        type=str,
        help="OpenAI model name for aggreation of sub-questions and CoT",
        default="gpt-4.1-2025-04-14",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["vilt", "blip"],
        help="VQA model type, either one of 'vilt' or 'blip",
    )
    parser.add_argument("--model_dir", type=str, default=None, help="Model directory")
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Directory containing the images"
    )

    args = parser.parse_args()

    client = None
    if args.prompting_mode in ["cot", "cot-consistent", "cot-sequential"]:
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("Error: OPENAI_API_KEY environment variable not set")
            sys.exit(1)
        client = OpenAI(api_key=openai_api_key)

    model_type_to_name = {
        "vilt": "dandelin/vilt-b32-mlm",
        "blip": "Salesforce/blip2-opt-2.7b",
    }

    model_mapping = {
        "vilt": VILTModel,
        "blip": BLIPModel,
    }

    model_dir = (
        args.model_dir
        if args.model_dir is not None
        else model_type_to_name[args.model_type]
    )
    model = model_mapping[args.model_type](model_dir=model_dir)

    json_results = eval_prompting(
        prompting_mode=args.prompting_mode,
        model=model,
        eval_dataset_path=args.eval_dataset_path,
        image_dir=args.image_dir,
        client=client,
        openai_model_name=args.openai_model_name,
        output_dir=args.output_dir,
    )

    write_json(args.output_dir, json_results)

    checkpoint_path = setup_checkpoint(args.output_dir)
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            logger.info(
                f"Checkpoint file {checkpoint_path} removed after successful completion"
            )
        except Exception as e:
            logger.error(f"Error removing checkpoint file: {e}")
