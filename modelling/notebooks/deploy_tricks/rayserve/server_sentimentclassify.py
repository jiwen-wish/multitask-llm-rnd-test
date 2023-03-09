from starlette.requests import Request
import ray
from ray import serve
from transformers import pipeline
from typing import List

@serve.deployment
class SentClassifier:
    def __init__(self):
        # Load model
        try:
            self.model = pipeline("sentiment-analysis", device=0)
        except:
            self.model = pipeline("sentiment-analysis")

    @serve.batch(max_batch_size=100)
    async def classify(self, inputs: List[Request]):
        # Run inference
        print("Our input array has length:", len(inputs))
        input_jsons = []
        for i in inputs:
            input_jsons.append(await i.json())
        print(input_jsons)
        return self.model([i['text'] for i in input_jsons])

    async def __call__(self, request: Request) -> dict:
        return await self.classify(request)
    
sentclassifier = SentClassifier.bind()
# curl -X POST -d '{"text":"Hello world!"}' http://localhost:8000