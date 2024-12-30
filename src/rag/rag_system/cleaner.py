import boto3
import yaml
from botocore.exceptions import ClientError
import click

def clean_endpoints(llm_endpoint_name:str, 
                    llm_model_name:str,
                    emb_endpoint_name: str,
                    emb_model_name:str
                    ):
    
    session = boto3.Session()

    sagemaker = session.client('sagemaker')

    try:
        llm_respone  = sagemaker.describe_endpoint(EndpointName=llm_endpoint_name)
        emb_respone  = sagemaker.describe_endpoint(EndpointName=emb_endpoint_name)
    except ClientError:
        print("Endpoints not found on AWS")
        return None 
        
    sagemaker.delete_endpoint(EndpointName=llm_endpoint_name)
    sagemaker.delete_model(ModelName=llm_model_name)
    sagemaker.delete_endpoint_config(EndpointConfigName=llm_endpoint_name)

    sagemaker.delete_endpoint(EndpointName=emb_endpoint_name)
    sagemaker.delete_model(ModelName=emb_model_name)
    sagemaker.delete_endpoint_config(EndpointConfigName=emb_endpoint_name)


    click.echo("Endpoints Successfully Deleted")



if __name__  == "__main__":
    with open("..\\..\\configs\\rag.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    EMB_ENDPOINT_NAME= cfg['EMB_ENDPOINT_NAME']
    EMB_MODEL_NAME =cfg['EMB_MODEL_NAME']
    LLM_ENDPOINT_NAME= cfg['LLM_ENDPOINT_NAME']
    LLM_MODEL_NAME = cfg['LLM_MODEL_NAME']
    clean_endpoints(
        llm_endpoint_name=LLM_ENDPOINT_NAME,
        llm_model_name=LLM_MODEL_NAME,
        emb_endpoint_name=EMB_ENDPOINT_NAME,
        emb_model_name=EMB_MODEL_NAME
    )