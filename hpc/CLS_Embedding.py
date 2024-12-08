from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModel,CLIPVisionModelWithProjection
import torch
from datasets import load_dataset
import datasets
import numpy as np
from datetime import datetime
import os
import sys
# import dill
# import multiprocessing
cpuCount = os.cpu_count()
print("Number of CPUs in the system:", cpuCount)
print(f'{datetime.now().strftime("%Y-%m-%d, %H:%M:%S")} Starts: Dataset Downloading')
dataset = load_dataset("MissTiny/WikiArt",cache_dir="/scratch/jy4057/MLProject")
print(f'{datetime.now().strftime("%Y-%m-%d, %H:%M:%S")} Ends: Dataset Finished')

print(f'{datetime.now().strftime("%Y-%m-%d, %H:%M:%S")} Starts: Model Loading')
model1 = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
processor1 = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
print(f'{datetime.now().strftime("%Y-%m-%d, %H:%M:%S")} Ends: Model Loaded')
def image_embedding(example, processor,model,torch):
    inputs = processor(images=example["image_numpy"], return_tensors="pt")
    outputs = model(**inputs)
    example["CLIPVisionModelWithProjection_image_embeds"] = torch.squeeze(outputs.image_embeds,0)
    # The CLIP Image embedding with size [512]
    return example

dataset.set_format(type="np", columns=['image_numpy'])
dataset.set_format(type="np", columns=['CLIPVisionModelWithProjection_image_embeds'])

# def image_cls_hidden_layer_embedding(example, processor,model,torch):
#     inputs = processor(images=example["image_numpy"], return_tensors="pt")
#     outputs = model(**inputs)
#     example["CLIPVisionModel_hidden_state"] = torch.squeeze(outputs.last_hidden_state,0)
#     example["CLIPVisionModel_CLS"] =  torch.squeeze(outputs.pooler_output,0)
    
#     # The CLIP Image embedding with size [512]
#     return example
def image_cls_hidden_layer_embedding(example):
    inputs = processor1(images=example["image_numpy"], return_tensors="pt")
    outputs = model1(**inputs)
    example["CLIPVisionModel_hidden_state"] = torch.squeeze(outputs.last_hidden_state,0)
    example["CLIPVisionModel_CLS"] =  torch.squeeze(outputs.pooler_output,0)
    
    # The CLIP Image embedding with size [512]
    return example
print(f'{datetime.now().strftime("%Y-%m-%d, %H:%M:%S")} Starts: Mapping')
# new_dataset = dataset.map(image_cls_hidden_layer_embedding,fn_kwargs={"processor": processor1, "model": model1,"torch":torch},num_proc = 4)
new_dataset = dataset.map(image_cls_hidden_layer_embedding,num_proc=8)
print(f'{datetime.now().strftime("%Y-%m-%d, %H:%M:%S")} Ends: Mapping')

from huggingface_hub import login
login(token="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")
print(f'{datetime.now().strftime("%Y-%m-%d, %H:%M:%S")} Starts: Uploading')
new_dataset.push_to_hub("MissTiny/WikiArt")
print(f'{datetime.now().strftime("%Y-%m-%d, %H:%M:%S")} All Done')