from ..rag_system.checker import check_endpoints
import yaml

with open("..\\..\\configs\\rag.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

LLM_ENDPOINT_NAME= cfg['LLM_ENDPOINT_NAME']
EMB_ENDPOINT_NAME= cfg['EMB_ENDPOINT_NAME']

def check_cmd():
    """Function for CLI Command that checks the endpoints on AWS"""
    check_endpoints(
    llm_endpoint_name=LLM_ENDPOINT_NAME,
    emb_endpoint_name=EMB_ENDPOINT_NAME,
    )