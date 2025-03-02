import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    SM_EXECUTION_ROLE_ARN = os.environ["SM_EXECUTION_ROLE_ARN"]
    SM_BUCKET = os.environ["SM_BUCKET"]
    SM_INFERENCE_IMAGE = os.environ["SM_INFERENCE_IMAGE"]
