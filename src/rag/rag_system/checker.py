import boto3
from botocore.exceptions import ClientError

def check_endpoints(llm_endpoint_name:str, 
                    emb_endpoint_name: str,
                    )-> None:
    session = boto3.Session()

    sagemaker = session.client('sagemaker')


    llm_status = "Found"
    emb_status = "Found"
    try:
        llm_respone  = sagemaker.describe_endpoint(EndpointName=llm_endpoint_name)
    except ClientError:
        llm_status = "Not Found"

    try:
        emb_respone  = sagemaker.describe_endpoint(EndpointName=emb_endpoint_name)
    except ClientError:
        emb_status = "Not Found"


    print(f"LLM Model Endpoint: {llm_status}")    
    print(f"Embedding Model Endpoint: {emb_status}")    
        

    