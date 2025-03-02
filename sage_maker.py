import json
from typing import Any

import boto3
from loguru import logger
from sagemaker.base_predictor import Predictor
from sagemaker.huggingface.llm_utils import get_huggingface_llm_image_uri
from sagemaker.huggingface.model import HuggingFaceModel

from config import Config


class SageMaker:

    def __init__(self):
        self.sm_client = boto3.client("sagemaker")

    def create_model(self, model_name: str, model_data_url: str) -> dict[str, Any]:
        response = self.sm_client.create_model(
            ModelName=model_name,
            ExecutionRoleArn=Config.SM_EXECUTION_ROLE_ARN,
            PrimaryContainer={
                "Image": Config.SM_INFERENCE_IMAGE,
                "ModelDataUrl": model_data_url,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                },
            },
        )

        return response

    @staticmethod
    def deploy_model(model_name: str, instance_type: str = "ml.g6.12xlarge") -> str:
        # Hub Model configuration. https://huggingface.co/models
        hub = {
            "HF_MODEL_ID": model_name,
            "SM_NUM_GPUS": json.dumps(4)
        }

        # create Hugging Face Model Class
        huggingface_model = HuggingFaceModel(
            image_uri=get_huggingface_llm_image_uri("huggingface", version="3.0.1"),
            env=hub,
            role=Config.SM_EXECUTION_ROLE_ARN,
        )

        # deploy model to SageMaker Inference
        predictor = huggingface_model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            container_startup_health_check_timeout=1200,
        )

        logger.info(f"Model deployed at endpoint: {predictor.endpoint_name}")
        return predictor.endpoint_name

    @staticmethod
    def predict(endpoint_name: str, inputs: str) -> object:
        predictor = Predictor(endpoint_name)
        return predictor.predict({
            "inputs": inputs,
        })

    @staticmethod
    def undeploy_model(endpoint_name: str) -> bool:
        predictor = Predictor(endpoint_name)
        predictor.delete_endpoint()
        return True
