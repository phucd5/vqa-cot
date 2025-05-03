import os
import torch
import logging
from dotenv import load_dotenv

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Blip2ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from huggingface_hub import HfFolder
from datasets import load_dataset
from PIL import Image

from model.base_model import Model
from utils.utils import get_device
from data.data_preprocess import preprocess_data_generation

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
logger = logging.getLogger("blip_model")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BLIPModel(Model):
    """
    BLIP Model class
    """

    def __init__(
        self,
        model_dir: str = "Salesforce/blip2-opt-2.7b",
        base_model: str = "Salesforce/blip2-opt-2.7b",
        train_data_path: str = None,
        val_data_path: str = None,
        output_dir: str = "../models/blip-finetuned",
        image_dir: str = "../data/images/images/",
    ):
        """
        Init the BLIP Model, if ft_data is provided, it will load the data with the model for fine-tuning
        Else, the model can only be used for inference (not training)

        Args:
            model_dir (str): name of the model to be used (compatible with HF)
            data_path (str): path to data containing json of (question_id, question, answer, image_path, full_answer)
            output_dir (str): output directory for saving the model
            image_dir (str): directory containing the images
        """
        super().__init__()
        self.model_dir = model_dir
        self.base_model = base_model
        self.output_dir = output_dir
        self.image_dir = image_dir
        self.data, self.validation_data = None, None

        self.processor = AutoProcessor.from_pretrained(self.base_model, use_fast=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        if train_data_path and val_data_path:
            self.load_data_with_model(train_data_path, val_data_path)
        else:
            self.model = Blip2ForConditionalGeneration.from_pretrained(self.model_dir)
        logger.info(
            f"Initialized model: {self.model_dir} w/ base_model {self.base_model}"
        )

        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )

    def load_data_with_model(self, train_data_path: str, val_data_path: str):
        """
        Load the data with the model for fine-tuning
        Args:
            train_data_path (str): train data path for json of data containing (question_id, question, answer, image_path, full_answer)
            val_data_path (str): train data path for json of data containing (question_id, question, answer, image_path, full_answer)
        """
        train_data = load_dataset("json", data_files=train_data_path)["train"]
        val_data = load_dataset("json", data_files=val_data_path)["train"]

        logger.info("Processing data")

        datasets = [(train_data, "train"), (val_data, "val")]

        for data, data_type in datasets:
            processed_data = data.map(
                preprocess_data_generation,
                batched=True,
                fn_kwargs={
                    "processor": self.processor,
                    "image_dir": self.image_dir,
                },
                keep_in_memory=True,
                remove_columns=[
                    "question_id",
                    "question",
                    "answer",
                    "image_path",
                    "full_answer",
                ],
            )
            if data_type == "train":
                self.data = processed_data
            elif data_type == "val":
                self.val_data = processed_data
        logger.info("Data processed")

        self.model = Blip2ForConditionalGeneration.from_pretrained(self.model_dir)

    def train(self, save_to_hf: bool = False):
        """
        Train the model with the data and save the model to Hugging Face Hub if enabled

        Args:
            save_to_hf (bool): whether to save the model to Hugging Face Hub
        """
        if not self.data:
            logger.error(
                "No data loaded for training, please run load_data_with_model()"
            )
            raise ValueError(
                "No data loaded for training, please run load_data_with_mdoel()"
            )

        try:
            if save_to_hf:
                HfFolder.save_token(HF_API_KEY)
        except Exception as e:
            logger.error(f"Error enabling save to Hugging Face repo {str(e)}")

        device = get_device()
        self.model.to(device)

        self.model.gradient_checkpointing_enable()
        self.model.language_model.gradient_checkpointing_enable()
        self.model.vision_model.gradient_checkpointing_enable()

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=5e-5,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_strategy="steps",
            logging_steps=50,
            fp16=True,
            remove_unused_columns=False,
            predict_with_generate=True,
            push_to_hub=save_to_hf,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.data,
            eval_dataset=self.val_data,
            data_collator=self.data_collator,
        )

        logger.info(f"Starting training of {self.model_dir} w/ device {device}")
        trainer.train()
        logger.info(f"Training finished")

        if save_to_hf:
            logger.info("Saving to Hugging Face Hub")
            try:
                trainer.push_to_hub()
                logger.info(f"Model pushed to Hub")
            except Exception as e:
                logger.error(f"Error pushing to Hub: {str(e)}")

    def forward(self, image_path: str, question: str):
        """
        Run inference on the model with the given image and question

        Args:
            image_path (str): path to image
            question (str): question to be answered

        Returns:
            str: predicted answer
        """
        device = get_device()
        self.model.to(device)
        self.model.eval()

        logger.info(
            f"Running inference on {image_path} with question: {question} w/ device {device}"
        )

        max_src_length = min(self.processor.tokenizer.model_max_length, 384)
        image = Image.open(image_path).convert("RGB")
        prompt = f"Question: {question} Answer:"
        inputs = self.processor(
            image,
            text=prompt,
            truncation=True,
            max_length=max_src_length,
            return_tensors="pt",
        ).to(device)

        # forward pass
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs)

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        return generated_text
