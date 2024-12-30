from sagemaker.huggingface import get_huggingface_llm_image_uri, HuggingFaceModel
import logging

# Get the logger for 'sagemaker.config'
sagemaker_config_logger = logging.getLogger()

# Set its level to WARNING or higher
sagemaker_config_logger.setLevel(logging.CRITICAL)