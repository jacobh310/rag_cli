import os
import json
from pinecone import Pinecone
import yaml
from typing import List
import boto3
from botocore.exceptions import ClientError
import numpy as np
from dotenv import load_dotenv, find_dotenv


# iam = boto3.client('iam')
# role = iam.get_role(RoleName='SageMakerExecutionRag')['Role']['Arn']

# sagemaker_runtime = boto3.client('sagemaker-runtime')

# pinecone_api_key = os.environ.get("PINECONE_API_KEY")
# # configure client
# pc = Pinecone(api_key=pinecone_api_key)

# # connect to index
# index = pc.Index(INDEX_NAME)


class Rag:
    """Class for Retrieval Augmented Generation System"""
    def __init__(self,
                 embed_model_endpoint_name,
                 llm_model_endpoint_name,
                 index_name
                 ) -> None:
        self.embed_model_endpoint_name = embed_model_endpoint_name
        self.llm_model_endpoint_name = llm_model_endpoint_name
        self.index_name = index_name
        self._set_up_cloud()

    def _set_up_cloud(self):
        
        iam = boto3.client('iam')
        self.role = iam.get_role(RoleName='SageMakerExecutionRag')['Role']['Arn']

        self.sagemaker_runtime = boto3.client('sagemaker-runtime')

        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        # configure client
        pc = Pinecone(api_key=pinecone_api_key)
        # connect to index
        self.index = pc.Index(self.index_name)

    def _check_endpoints(self):
        session = boto3.Session()
        sagemaker = session.client('sagemaker')

        try:
            emb_respone  = sagemaker.describe_endpoint(EndpointName=self.embed_model_endpoint_name)
            llm_respone  = sagemaker.describe_endpoint(EndpointName=self.llm_model_endpoint_name)
        except ClientError:
            print("End point not found on AWS. Run deploy.py")
            return False

        return True


    
    def _invoke_embed_model(self,
                           strings: List):
        """
        Takes a list of strings and returns embedding matrix for each string
        """
        payload = {
        "inputs": strings }

        payload = json.dumps(payload)

        response = self.sagemaker_runtime.invoke_endpoint(
            EndpointName=self.embed_model_endpoint_name,
            ContentType='application/json',
            Accept='application/json',
            Body=payload
        )

        response_body = response['Body'].read()
        result = json.loads(response_body)

        embeddings = np.mean(np.array(result), axis=1)
        return embeddings.tolist()[0]

    
    
    def _get_embeds(self,
                    query_vec,
                    filter):
        
        vec_embeds = self.index.query(vector=query_vec, top_k=5, filter=filter, include_metadata=True)
        return vec_embeds
    
    def _construct_context(self,
                           vec_embeds,
                          max_section_len: int ) -> str:
        
        contexts = [match.metadata["text"] for match in vec_embeds.matches]
        chosen_sections = []
        chosen_sections_len = 0
        separator = "\n"


        for text in contexts:
            text = text.strip()
            # Add contexts until we run out of space.
            chosen_sections_len += len(text) + 2
            if chosen_sections_len > max_section_len:
                break
            chosen_sections.append(text)
        concatenated_doc = separator.join(chosen_sections)
        return concatenated_doc

    
    def _create_rag_payload(self,
            prompt,
              context_str) -> dict:
        
        prompt_template = """
        You are an expert in finance who is ready for question answering tasks. Use the context below to answer the question. Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. Use five sentences maximum and keep the answer concise.
    
        Context: {context}
        
        Question: {prompt}
        
        Answer:
        """

        text_input = prompt_template.replace("{context}", context_str).replace("{prompt}", prompt)

        payload = {
            "inputs":  f"System: {text_input}\nUser: {prompt}",
            "parameters":{
                        "max_new_tokens": 512, 
                        "top_p": 0.9, 
                        "temperature": 0.6, 
                        "return_full_text": False}
        }

        payload = json.dumps(payload)
        
        return(payload)

    
    
    def _invoke_llm_model(self,
                          payload):
        # Invoke the SageMaker endpoint
        response = self.sagemaker_runtime.invoke_endpoint(
            EndpointName= self.llm_model_endpoint_name,
            ContentType='application/json',
            Accept='application/json',
            Body=payload
        )

        response_body = response['Body'].read()
        result = json.loads(response_body)

        return result[0]['generated_text']
    
    def rag_query(self,
                  prompt,
                  filter):
        """Processes the query and returns the response from LLM model"""
        endpoint = self._check_endpoints()

        if not endpoint:
            return
        else: 
            query_vec = self._invoke_embed_model(prompt)
            vec_embeds = self._get_embeds(query_vec=query_vec,
                                            filter=filter)
            
            context = self._construct_context(vec_embeds=vec_embeds,
                                                max_section_len=2500)
            
            payload = self._create_rag_payload(prompt=prompt,
                                        context_str=context)
            
            response = self._invoke_llm_model(payload=payload)
            
            return response
    
    def chat(self):
        """Chat with the RAG system"""
        
        while True:
            filter_input = input("Company Ticker: ")
            prompt = input("Question: ")
            filter = {"Company": {"$eq":filter_input.upper()}}
            
            if prompt.lower().strip() or filter_input.lower().strip() == "exit":
                break
            response = self.rag_query(
                        prompt,
                        filter=filter
                        )
            print(response)
        

