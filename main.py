import os
import tempfile

from loguru import logger

from config import Config
from huggingface import HuggingFace
from s3 import S3
from sage_maker import SageMaker
from utils import get_model_dir_path


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


def save_model_to_disk(model_name: str, dir_path: str):
    logger.info(f"Save model '{model_name}' to '{dir_path}' - start")
    HuggingFace.save_model(model_name, dir_path)
    logger.debug(f"Weightings for model '{model_name}' saved to '{dir_path}' - start")
    HuggingFace.save_tokenizer(model_name, dir_path)
    logger.debug(f"Tokenizer for model '{model_name}' saved to '{dir_path}' - start")
    logger.info(f"Save model '{model_name}' to '{dir_path}' - done")


def upload_model_from_disk(model_name: str, dir_path: str):
    logger.info(f"Upload model '{model_name}' from '{dir_path}' - start")
    s3 = S3()
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            s3_key = f"{get_model_dir_path(model_name)}/{file}"
            s3.upload_file(file_path, Config.SM_BUCKET, s3_key)
    logger.info(f"Upload model '{model_name}' from '{dir_path}' - done")


def sagemaker_create_model(model_name: str):
    response = SageMaker().create_model(model_name, get_model_dir_path(model_name))
    print(response)


def main(func_name, model_name: str, *args, **kwargs):
    func_map = {
        "save_and_upload_model_via_tempdir": save_and_upload_model_via_tempdir,
        "save_model_to_disk": save_model_to_disk,
        "upload_model_from_disk": upload_model_from_disk,
        "sagemaker_create_model": sagemaker_create_model,
    }
    return func_map[func_name](model_name, *args, **kwargs)


if __name__ == "__main__":
    FUNC = "save_model_to_disk"
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    kwargs = {"dir_path": "/Users/davidmcelhill/Documents/models"}
    main(FUNC, MODEL_NAME, **kwargs)
