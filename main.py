import argparse
import os
import tempfile

from loguru import logger

from config import Config
from huggingface import HuggingFace
from s3 import S3
from sage_maker import SageMaker
from utils import get_model_dir_path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def save_and_upload_model_via_tempdir(model_name: str):
    logger.info(f"Save and upload model '{model_name}' via tempdir - start")
    s3 = S3()
    with tempfile.TemporaryDirectory() as temp_dir:
        HuggingFace.save_model(model_name, temp_dir)
        HuggingFace.save_tokenizer(model_name, temp_dir)

        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                s3_key = f"{get_model_dir_path(model_name)}/{file}"
                s3.upload_file(file_path, Config.SM_BUCKET, s3_key)

    logger.info(f"Save and upload model '{model_name}' via tempdir - done")


def save_model_to_disk(model_name: str, models_dir: str = MODELS_DIR):
    """Save a model files into a model-named sub-directory within the models directory"""
    model_dir = os.path.join(models_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Save model '{model_name}' to '{model_dir}' - start")
    HuggingFace.save_model(model_name, model_dir)
    logger.debug(f"Weightings for model '{model_name}' saved to '{model_dir}' - start")
    HuggingFace.save_tokenizer(model_name, model_dir)
    logger.debug(f"Tokenizer for model '{model_name}' saved to '{model_dir}' - start")
    logger.info(f"Save model '{model_name}' to '{model_dir}' - done")


def upload_model_from_disk(model_name: str, model_dir: str):
    logger.info(f"Upload model '{model_name}' from '{model_dir}' - start")
    s3 = S3()
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            file_path = os.path.join(root, file)
            s3_key = f"{get_model_dir_path(model_name)}/{file}"
            s3.upload_file(file_path, Config.SM_BUCKET, s3_key)
    logger.info(f"Upload model '{model_name}' from '{model_dir}' - done")


def sagemaker_create_model(model_name: str):
    response = SageMaker().create_model(model_name, get_model_dir_path(model_name))
    print(response)


def main(func_name, model_name: str, models_dir: str = None, model_dir: str = None):
    func_map = {
        "save_and_upload_model_via_tempdir": save_and_upload_model_via_tempdir,
        "save_model_to_disk": save_model_to_disk,
        "upload_model_from_disk": upload_model_from_disk,
        "sagemaker_create_model": sagemaker_create_model,
    }

    if func_name not in func_map:
        raise ValueError(f"function name '{func_name}' is not recognised")

    func_map[func_name](model_name, models_dir=models_dir, model_dir=model_dir)


if __name__ == "__main__":
    # python main.py save_model_to_disk deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
    parser = argparse.ArgumentParser(description="CLI for invoking main with a function name and optional arguments.")

    parser.add_argument("func_name", type=str, help="Name of the function to execute")
    parser.add_argument(
        "model_name",
        type=str,
        help="Full name of the model on HuggingFace, e.g. deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    )
    parser.add_argument("--models_dir", type=str, help="Optional 'models' directory path where all models are stored")
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Optional 'model' directory path where the files can be found for a specific model"
    )

    args = parser.parse_args()

    main(args.func_name, args.model_name, args.models_dir, args.model_dir)
