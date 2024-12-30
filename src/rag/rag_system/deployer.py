import os
import json

import boto3
from botocore.exceptions import ClientError
from sagemaker.huggingface import get_huggingface_llm_image_uri, HuggingFaceModel

from dotenv import load_dotenv, find_dotenv

import yaml




class Deployer:
    """Initialize Deployer Class"""

    def __init__(self,
                 instance_type,
                 initial_instnace_count,
                 health_checkout):
        self.instance_type = instance_type
        self.initial_instnace_count = initial_instnace_count
        self.health_checkout = health_checkout


    def launch_endpoint(self,
                        role,
                        model_name,
                        model_config_dict,
                        image_name=None,
                        image_version=None,
                        **model_configs):
         
         endpoint = self._check_endpoints(model_configs.get('endpoint_name'))

         if endpoint:
            return
         
         if image_name:
                
            image_uri = get_huggingface_llm_image_uri(image_name,
                                              version = image_version)
         else:
             image_uri = None 

         model = HuggingFaceModel(
            name= model_name,
            env=model_config_dict,
            role=role,
            transformers_version= model_configs.get('transformers_version') ,  # transformers version used
            pytorch_version= model_configs.get('pytorch_version'),  # pytorch version used
            py_version=model_configs.get('py_version'), # python version of the DLC
            image_uri =image_uri

            )

         model.deploy(
                    initial_instance_count= model_configs.get('initial_instance_count'),
                    instance_type=model_configs.get('instance_type'),
                    endpoint_name=model_configs.get('endpoint_name')
                    )

         print(f"Launched Model {model_name}")

    def _check_endpoints(self, endpoint_name):
        session = boto3.Session()
        sagemaker = session.client('sagemaker')

        try:
            respone  = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        except ClientError:
            
            return False
        print("Endpoint already exists on AWS")
        return True

if __name__  == "__main__":



    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    with open("..\\..\\configs\\rag.yaml", "r") as f:
        
        cfg = yaml.load(f, Loader=yaml.FullLoader)


    # Environment Variable
    HF_TOKEN = os.environ.get("HUGGING_FACE_AUTH")
    os.environ['AWS_ACCESS_KEY_ID'] = os.environ.get("AWS_ACCESS_KEY")
    os.environ['AWS_SECRET_ACCESS_KEY']= os.environ.get("AWS_SECRET_ACCESS_KEY")
    os.environ['AWS_DEFAULT_REGION'] = os.environ.get('AWS_DEFAULT_REGION')


    # Mini LM Configuaration rag.yaml
    ROLE = cfg['ROLE']
    EMB_MODEL_NAME = cfg['EMB_MODEL_NAME']
    EMB_INSTNACE_TYPE= cfg['EMB_INSTNACE_TYPE']
    EMB_INITIAL_INSTANCE_COUNT=cfg['EMB_INITIAL_INSTANCE_COUNT']
    EMB_HEALTH_CHECK_TIMEOUT= cfg['EMB_HEALTH_CHECK_TIMEOUT']
    EMB_ENDPOINT_NAME= cfg['EMB_ENDPOINT_NAME']
    EMB_MODEL_ID = cfg['EMB_MODEL_ID']

    # LLM Configurations
    LLM_MODEL_NAME = cfg['LLM_MODEL_NAME']
    LLM_INSTNACE_TYPE= cfg['LLM_INSTNACE_TYPE']
    LLM_INITIAL_INSTANCE_COUNT=cfg['LLM_INITIAL_INSTANCE_COUNT']
    LLM_HEALTH_CHECK_TIMEOUT= cfg['LLM_HEALTH_CHECK_TIMEOUT']
    LLM_ENDPOINT_NAME= cfg['LLM_ENDPOINT_NAME']
    LLM_MODEL_ID = cfg['LLM_MODEL_ID']
    LLM_NUMBER_OF_GPUS = cfg['LLM_NUMBER_OF_GPUS']
    LLL_MAX_INPUT_LENGTH= cfg['LLL_MAX_INPUT_LENGTH']
    LLM_MAX_TOTAL_TOKENS= cfg['LLM_MAX_TOTAL_TOKENS']
    LLM_MAX_BATCH_TOTAL_TOKENS = cfg['LLM_MAX_BATCH_TOTAL_TOKENS']

    iam = boto3.client('iam')
    role = iam.get_role(RoleName=ROLE)['Role']['Arn']

    mini_lm_configs =  {
    "HF_MODEL_ID":EMB_MODEL_ID,  # model_id from hf.co/models
    "HF_TASK": "feature-extraction",
        }


    llm_configs = {
    'HF_MODEL_ID': LLM_MODEL_ID, # model_id from hf.co/models
    'SM_NUM_GPUS': json.dumps(LLM_NUMBER_OF_GPUS), # Number of GPU used per replica
    'MAX_INPUT_LENGTH': json.dumps(LLL_MAX_INPUT_LENGTH),  # Max length of input text
    'MAX_TOTAL_TOKENS': json.dumps(LLM_MAX_TOTAL_TOKENS),  # Max length of the generation (including input text)
    'MAX_BATCH_TOTAL_TOKENS': json.dumps(LLM_MAX_BATCH_TOTAL_TOKENS),  # Limits the number of tokens that can be processed in parallel during the generation
    'HUGGING_FACE_HUB_TOKEN': HF_TOKEN,
    'HF_MODEL_QUANTIZE': "bitsandbytes", # comment in to quantize
    }


    mini_lm = Deployer(
        instance_type=EMB_INSTNACE_TYPE,
        initial_instnace_count= EMB_INITIAL_INSTANCE_COUNT,
        health_checkout = EMB_HEALTH_CHECK_TIMEOUT
                            )

    mini_lm.launch_endpoint(
        role = role,
        model_name = EMB_MODEL_NAME,
        model_config_dict= mini_lm_configs,
        transformers_version="4.6",  # transformers version used
        pytorch_version="1.7",  # pytorch version used
        py_version="py36", 
        initial_instance_count = EMB_INITIAL_INSTANCE_COUNT,
        instance_type= EMB_INSTNACE_TYPE,
        endpoint_name= EMB_ENDPOINT_NAME
    )
    llama = Deployer(
            instance_type=LLM_INSTNACE_TYPE,
            initial_instnace_count= LLM_INITIAL_INSTANCE_COUNT,
            health_checkout = LLM_HEALTH_CHECK_TIMEOUT
            )

    llama.launch_endpoint(
        image_name = 'huggingface',
        image_version= "2.0.1",
        role = role,
        model_name = LLM_MODEL_NAME,
        model_config_dict= llm_configs,
        initial_instance_count = LLM_INITIAL_INSTANCE_COUNT,
        instance_type= LLM_INSTNACE_TYPE,
        endpoint_name= LLM_ENDPOINT_NAME
    )
