import argparse

from model.blip_model import BLIPModel
from model.vilt_model import VILTModel
from utils.utils import setup_logging


if __name__ == "__main__":
    setup_logging("logs/model_vilt_train.log")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="vilt-finetuned",
        type=str,
        help="Output directory for the model",
    )
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Directory containing the images"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["vilt", "blip"],
        help="VQA model type, either one of 'vilt' or 'blip",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Directory for the preprocessed validation data json",
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        required=True,
        help="Directory for the preprocessed train data json",
    )
    parser.add_argument(
        "--model_dir", type=str, default=None, help="custom model directory"
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default=None,
        help="base model name for the model for processor (e.g, dandelin/vilt-b32-mlm)",
    )
    args = parser.parse_args()

    # use also for base model
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
    base_model = (
        args.base_model_name
        if args.base_model_name is not None
        else model_type_to_name[args.model_type]
    )

    model = model_mapping[args.model_type](
        model_dir=model_dir,
        base_model=base_model,
        train_data_path=args.train_data_dir,
        val_data_path=args.val_data_dir,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
    )

    model.train(save_to_hf=True)
