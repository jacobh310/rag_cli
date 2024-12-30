import boto3
import yaml
from pinecone import Pinecone
import os

with open("..\\..\\configs\\rag.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
