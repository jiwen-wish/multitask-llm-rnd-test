from pinferencia import Server


from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModel

model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval()
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt").to('cuda')

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooler_output


def predict(data):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(images=image, return_tensors="pt").to('cuda')

    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    pooled_output = outputs.pooler_output
    return "dummy"


service = Server()
service.register(model_name="mymodel", model=predict)