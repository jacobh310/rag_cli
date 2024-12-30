from sec_edgar_downloader import Downloader
import yaml
from bs4 import BeautifulSoup
from dotenv import load_dotenv, find_dotenv
from utils import get_all_file_paths
import os 

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

with open("..\\..\\configs\\data.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

path = '../../data/raw'
EMAIL = os.environ.get("EMAIL")
TICKERS = cfg['tickers']
AMOUNT = cfg['amount'] 

dl = Downloader("RAG", EMAIL, path,)

for ticker in TICKERS:
    dl.get("10-K", ticker, limit= AMOUNT)

print("Finish Downloading")

directory_path = "..\..\data\\raw\sec-edgar-filings"
file_paths = get_all_file_paths(directory_path)

print("Parsing HTML and Saving")

for path in file_paths:
    
    file = open(path, 'r', encoding='utf-8')
    soup = BeautifulSoup(file, 'html.parser')
    text = soup.sequence
    file.close()
    with open(path,"w",encoding='utf-8',errors='ignore') as file:
        file.write(str(text))

print("Finished")