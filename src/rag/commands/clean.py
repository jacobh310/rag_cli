import yaml
from ..rag_system.cleaner import clean_endpoints

with open("..\\..\\configs\\rag.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

EMB_ENDPOINT_NAME= cfg['EMB_ENDPOINT_NAME']
EMB_MODEL_NAME =cfg['EMB_MODEL_NAME']
LLM_ENDPOINT_NAME= cfg['LLM_ENDPOINT_NAME']
LLM_MODEL_NAME = cfg['LLM_MODEL_NAME']


def clean_cmd():
    """Function for CLI Command that cleans the endpoints on AWS"""

    clean_endpoints(
    llm_endpoint_name=LLM_ENDPOINT_NAME,
    llm_model_name=LLM_MODEL_NAME,
    emb_endpoint_name=EMB_ENDPOINT_NAME,
    emb_model_name=EMB_MODEL_NAME
    )