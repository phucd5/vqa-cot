import logging
import os
import re
import sys
from textwrap import dedent
from typing import List, Tuple

import openai
from dotenv import load_dotenv
from openai import OpenAI

from model.base_model import Model
from model.blip_model import BLIPModel
from model.vilt_model import VILTModel
from utils.utils import setup_logging

logger = logging.getLogger("llm_prompt")


def extract_blip_answer(answer_txt: str) -> str:
    """Extracts the answer from the input string after the 'Answer:' keyword."""
    if "answer:" in answer_txt.lower():
        answer_part = answer_txt.lower().split("answer:")[1].strip()
        return answer_part
    else:
        return answer_txt.strip()


def extract_sub_questions(text: str) -> List[str]:
    """
    Extracts sub-questions from lines beginning with "SubQ".
    Matches e.g.:
      SubQ: …
      SubQ1: …
      SubQ42: …
    """
    pattern = re.compile(r"^\s*SubQ\d*:\s*(.+)$")

    subqs = []
    for line in text.splitlines():
        m = pattern.match(line)
        if m:
            subqs.append(m.group(1).strip())
    logger.info(f"Extracted sub-questions: {subqs}")
    return subqs


def extract_final_answer(text: str) -> str:
    """
    Extracts the final answer from the given text
    Args:
        text (str): Input text containing the final answer
    Returns:
        str: Extracted final answer
    """
    parts = text.split("Answer:", 1)
    if len(parts) < 2:
        return None
    return parts[1].strip()


def load_sys_prompt(file_path: str) -> str:
    """
    Loads the system prompt from a file
    Args:
        file_path (str): Path to the prompt file
    Returns:
        str: system prompt
    """
    prompt = None
    with open(file_path, "r") as f:
        prompt = f.read()

    prompt.strip()
    prompt = dedent(prompt)
    return prompt


def load_prompt(file_path: str, question: str) -> str:
    """
    Loads the prompt from a file and replaces placeholders with actual values
    Args:
        file_path (str): The path to the prompt file
        question (str): The complex question to be inserted
    Returns:
        str: formatted prompt
    """
    prompt = None
    with open(file_path, "r") as f:
        prompt = f.read()

    prompt = prompt.replace("[Insert Complex Question Here]", question)

    prompt.strip()
    prompt = dedent(prompt)
    return prompt


def load_final_answer_prompt(
    file_path: str, qa_pairs: List[Tuple[str, str]], question: str
) -> str:
    """
    Loads the final answer prompt from a file
    Args:
        file_path (str): path to the prompt file
        qa_pairs (list): list of sub-questions and answers
        question (str): complex question
        choices (list): list of answer choices
    Returns:
        str: formatted prompt
    """
    prompt = None
    with open(file_path, "r") as f:
        prompt = f.read()

    qa_str = ""
    for i, (sub_q, ans) in enumerate(qa_pairs):
        qa_str += f"Sub-Q{i+1}: {sub_q}\nAnswer: {ans}\n"

    prompt = prompt.replace("[Insert QA-Pairs Here]", qa_str)
    prompt = prompt.replace("[Insert Complex Question Here]", question)
    prompt.strip()
    prompt = dedent(prompt)
    return prompt


def get_sub_questions(
    model_name: str, client, question: str, model_type: str
) -> List[str]:
    """
    Generates sub-questions
    Args:
        model_name (str): name of the OpenAI model to use
        client (OpenAI): OpenAI client instance.
        question (str): audio-visual question
        model_type (str): type of sub-question generation (classification or generation)
    Returns:
        list: sub-questions
    """

    sys_prompt_file = (
        "cot/prompts/cot/sys_instr_gen.txt"
        if model_type == "generation"
        else "cot/prompts/cot/sys_instr_class.txt"
    )

    logger.info("---------------------")
    logger.info(
        f"Generating sub-questions for normal CoT: {question} w/ {model_type} and {model_name}"
    )
    sys_prompt = load_sys_prompt(sys_prompt_file)
    prompt = load_prompt("cot/prompts/cot/prompt.txt", question)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        res = response.choices[0].message.content.strip()

        logger.info(f"Model Output:\n {res}")
        logger.info("---------------------")
        return extract_sub_questions(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error getting k-sub questions {e}")
        return None


def get_sub_questions_sequential(
    vqa_model: Model,
    image_path: str,
    model_name: str,
    client,
    question: str,
    model_type: str = "generation",
    k: int = 5,
) -> List[Tuple[str, str]]:
    """
    Generates k sub-questions sequentially
    Args:
        model_name (str): name of the OpenAI model to use
        client (OpenAI): OpenAI client instance
        question (str): audio-visual question
        k (int): number of sub-questions to generate

    Returns:
        list: (sub-question, answer) pairs
    """
    sys_prompt_file = (
        "cot/prompts/cot_sequential/sys_instr_gen.txt"
        if model_type == "generation"
        else "cot/prompts/cot_sequential/sys_instr_class.txt"
    )
    qa_pairs = []
    sys_prompt = load_sys_prompt(sys_prompt_file)
    prompt = load_prompt(
        "cot/prompts/cot_sequential/prompt.txt",
        question,
        k=1,
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]

    logger.info(f"Generating {k} sub-questions for sequential CoT: {question}")

    for i in range(1, k + 1):
        logger.info(f"Generating sub-question {i} for sequential CoT")
        try:
            response = client.chat.completions.create(
                model=model_name, messages=messages
            )
            sub_q_res = response.choices[0].message.content.strip()

            messages.append(
                {
                    "role": "assistant",
                    "content": sub_q_res,
                }
            )
            sub_q = extract_sub_questions(sub_q_res)
            if len(sub_q) == 0:
                logger.info("No sub-questions generated")
                return None
            elif sub_q[0] == "DONE":
                logger.info("Sub-question generation completed")
                break

            sub_q = sub_q[0]
            answer = vqa_model.forward(image_path, sub_q)
            if isinstance(vqa_model, BLIPModel):
                answer = extract_blip_answer(answer)

            # feed back the sub-question and answer to the model
            messages.append(
                {
                    "role": "user",
                    "content": f"SubQ{i}: {sub_q}\nAnswer: {answer}",
                }
            )
            logger.info(f"VQA Model Output: {answer}")
            qa_pairs.append((sub_q, answer))
        except Exception as e:
            print(f"Error getting sub-questions sequentially {e}")
            return None
    logger.info("---------------------")
    return qa_pairs


def get_final_answer(
    model_name: str,
    client,
    qa_pairs: List[Tuple[str, str]],
    question: str,
):
    """
    Generates the final answer based on sub-questions

    Args:
        model_name (str): name of the OpenAI model to use
        client (OpenAI): OpenAI client instance
        qa_pairs (list): list of sub-questions and answers
        question (str): audio-visual question


    Returns:
        str: final answer
    """
    sys_prompt = load_sys_prompt("cot/prompts/cot_aggregation/sys_instr.txt")
    prompt = load_final_answer_prompt(
        "cot/prompts/cot_aggregation/prompt.txt", qa_pairs, question
    )
    logger.info("---------------------")
    logger.info(f"Generating final answer for: {question}")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        res = response.choices[0].message.content.strip()
        logger.info(f"Model Output: {res}")
        logger.info("---------------------")
        return extract_final_answer(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error getting final answer: {e}")
        return None


if __name__ == "__main__":
    # avoid circular import
    from eval import (
        predict_cot,
        predict_cot_consistent,
        predict_cot_sequential,
        predict_direct,
    )

    # example usage
    # run using python -m cot.llm_prompt
    setup_logging("logs/llm_test.log", level=logging.INFO)

    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    client = OpenAI(api_key=openai_api_key)

    model = BLIPModel()
    answer = predict_cot(
        model=model,
        question="What country is this picture taken in?",
        image_path="data/japan.jpg",
        client=client,
        openai_model_name="o4-mini-2025-04-16",
    )
    logger.info(f"Final Answer: {answer}")
