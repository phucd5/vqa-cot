import os
from dotenv import load_dotenv
import logging

from model.base_model import Model
from utils.utils import get_device
from data.data_preprocess import generate_label_mapping, preprocess_data_classification

import torch
from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers import TrainingArguments, Trainer, DefaultDataCollator
from huggingface_hub import HfFolder
from datasets import Dataset, load_dataset
from PIL import Image

load_dotenv()

# Constants
HF_API_KEY = os.getenv("HF_API_KEY")

logger = logging.getLogger("vilt_model")


class VILTModel(Model):
    """
    VILT Model class
    """

    def __init__(
        self,
        model_dir: str = "dandelin/vilt-b32-mlm",
        base_model: str = "dandelin/vilt-b32-mlm",
        train_data_path: str = None,
        val_data_path: str = None,
        output_dir: str = "../models/vilt-finetuned",
        image_dir: str = "../data/Images/images/",
    ):
        """
        Initalizes the VILT model, if ft_data is provided, it will load the data with the model for fine-tuning
        Else, the model can only be used for inference (not training)

        Args:
            model_dir (str): dir of the model to be used (can be custom dir or hugging face)
            base_model (str): name of the base model used for processor
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

        self.processor = ViltProcessor.from_pretrained(self.base_model)
        if train_data_path and val_data_path:
            self.load_data_with_model(train_data_path, val_data_path)
        else:
            self.model = ViltForQuestionAnswering.from_pretrained(self.model_dir)
        logger.info(
            f"Initialized model: {self.model_dir} w/ base_model {self.base_model}"
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

        label2id, id2label = generate_label_mapping([train_data, val_data])

        datasets = [(train_data, "train"), (val_data, "val")]

        for data, data_type in datasets:
            logger.info(f"Processing data {data_type} with len: {len(data)}")
            processed_data = data.map(
                preprocess_data_classification,
                batched=True,
                fn_kwargs={
                    "processor": self.processor,
                    "label2id": label2id,
                    "image_dir": self.image_dir,
                },
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

        self.model = ViltForQuestionAnswering.from_pretrained(
            self.model_dir,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
        )

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
                "No data loaded for training, please run load_data_with_model()"
            )

        try:
            if save_to_hf:
                HfFolder.save_token(HF_API_KEY)
        except Exception as e:
            logger.error(f"Error enabling save to Hugging Face repo {str(e)}")

        device = get_device()
        self.model.to(device)

        training_args = TrainingArguments(
            output_dir=f"{self.output_dir}",
            per_device_train_batch_size=16,
            num_train_epochs=20,
            save_steps=200,
            logging_steps=50,
            learning_rate=5e-5,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=save_to_hf,
            gradient_accumulation_steps=2,
            dataloader_num_workers=8,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=DefaultDataCollator(),
            train_dataset=self.data,
            eval_dataset=self.val_data,
        )

        logger.info(f"Starting training of {self.model_dir} w/ device {device}")
        trainer.train()
        logger.info(f"Training finished, attempt to save to {self.output_dir}")

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
            image_path (str): path to the image
            question (str): question to be asked
        Returns:
            str: predicted answer
        """
        device = get_device()
        self.model.to(device)
        self.model.eval()
        logger.info(
            f"Running inference on {image_path} with question: {question} w/ device {device}"
        )

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            images=image,
            text=question,
            truncation=True,
            max_length=self.processor.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(device)

        # forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        label_id = outputs.logits.argmax(-1).item()
        return self.model.config.id2label[label_id]
