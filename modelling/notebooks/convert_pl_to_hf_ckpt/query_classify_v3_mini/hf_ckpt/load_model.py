#%%
# transformers==4.22.2
# torch==1.12.1
# python 3.8.13

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import os

def load_pipeline():
    """
    pipe(['Classify query: apple'], batch_size=10)
    """
    file_path = os.path.dirname(__file__)
    model_hf_reload = AutoModelForSequenceClassification.from_pretrained(file_path).eval()
    tokenizer_hf_reload = AutoTokenizer.from_pretrained(file_path, fast=True)
    tokenizer_kwargs = {'padding':True, 'truncation':True, 'max_length':56}
    with torch.no_grad():
        model_hf_reload.bert.pooler.activation = torch.nn.Identity()
    try:
        pipe = pipeline("text-classification", model=model_hf_reload, 
            tokenizer=tokenizer_hf_reload, function_to_apply='sigmoid', device=0, 
            **tokenizer_kwargs)
    except Exception as e:
        print(f'use CPU instead due to {e}')
        pipe = pipeline("text-classification", model=model_hf_reload, 
            tokenizer=tokenizer_hf_reload, function_to_apply='sigmoid', device=-1,
            **tokenizer_kwargs)
    return pipe
# %%
