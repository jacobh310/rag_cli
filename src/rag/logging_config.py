import logging
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# Configure root logger
logging.getLogger().setLevel(logging.ERROR)

# Configure specific loggers
LOGGERS_TO_SILENCE = [
    'sagemaker',
    'sagemaker.config',
    'botocore',
    'boto3',
    'urllib3',
    'matplotlib',
    'fsspec',
    'aiobotocore',
    'asyncio'
]

for logger_name in LOGGERS_TO_SILENCE:
    logging.getLogger(logger_name).setLevel(logging.ERROR)
    logging.getLogger(logger_name).propagate = False 