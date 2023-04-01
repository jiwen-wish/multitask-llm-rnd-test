#%%
import requests 
import numpy as np

image_url = "https://canary.contestimg.wish.com/api/webimage/61b241a3a4ee2ecaf2f63c77-large.jpg?cache_buster=bbeee1fdb460a1d12bc266824914e030"

input_json = {
   "inputs":[
      {	
      "name": "image_url",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": [image_url]
      }
   ]
}

res = requests.post('http://localhost:8000/v2/models/clip_image_ensemble/versions/1/infer', json=input_json).json()

pooled_output_onnx = np.array(res['outputs'][0]['data']).reshape(res['outputs'][0]['shape'])
#%%
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = image_url
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")

outputs = model.get_image_features(**inputs)
pooled_output_hf = outputs.detach().cpu().numpy()
# %%
import matplotlib.pyplot as plt 
plt.hist(np.abs(pooled_output_onnx - pooled_output_hf).reshape(-1).tolist())
# %%
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
image = preprocess(Image.open(requests.get(url, stream=True).raw)).unsqueeze(0).to(device)


with torch.no_grad():
   image_features = model.encode_image(image)
pooled_output_clip = image_features.detach().cpu().numpy()

#%%
plt.hist(np.abs(pooled_output_clip - pooled_output_hf).reshape(-1).tolist())

# %%


url = "https://canary.contestimg.wish.com/api/webimage/61b241a3a4ee2ecaf2f63c77-large.jpg?cache_buster=bbeee1fdb460a1d12bc266824914e030"

# get HF image fearures
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")

outputs = model.get_image_features(**inputs)
pooled_output_hf = outputs.detach().cpu().numpy()

# get OpenAI image features
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
image = preprocess(Image.open(requests.get(url, stream=True).raw)).unsqueeze(0).to(device)


with torch.no_grad():
   image_features = model.encode_image(image)
pooled_output_clip = image_features.detach().cpu().numpy()

# check difference
assert np.allclose(pooled_output_hf, pooled_output_clip, atol=0.1), "hf and clip too different"
# %%
