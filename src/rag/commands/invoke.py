from ..rag_system.invoker import Rag
import yaml
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

with open("..\\..\\configs\\rag.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

EMB_ENDPOINT_NAME= cfg['EMB_ENDPOINT_NAME']
LLM_ENDPOINT_NAME= cfg['LLM_ENDPOINT_NAME']
INDEX_NAME = cfg['INDEX_NAME']

def invoke_cmd():
    """Function for CLI command to invoke RAG"""
    rag = Rag(
        embed_model_endpoint_name=EMB_ENDPOINT_NAME,
        llm_model_endpoint_name = LLM_ENDPOINT_NAME,
        index_name= INDEX_NAME
              )
    rag.chat()