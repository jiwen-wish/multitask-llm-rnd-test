from starlette.requests import Request
import ray
from ray import serve
from typing import List
from dvc.api import DVCFileSystem
import tempfile
import sys
import os
import importlib


repo = 'git@github.com:ContextLogic/multitask-llm-rnd.git'
path = '/modelling/notebooks/convert_pl_to_hf_ckpt/query_classify_v3_mini/hf_ckpt/'
max_batch_size = 100

def import_path(path):
    module_name = os.path.basename(path).replace('-', '_')
    spec = importlib.util.spec_from_loader(
        module_name,
        importlib.machinery.SourceFileLoader(module_name, path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module

@serve.deployment(ray_actor_options={"num_gpus": 1})
class QueryClassifier:
    def __init__(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            fs = DVCFileSystem(repo, subrepos=True)
            fs.get(path, tmpdirname, recursive=True)
            os.system(f'ls {tmpdirname}')
            self.model = import_path(f"{tmpdirname}/load_model.py").load_pipeline()

    @serve.batch(max_batch_size=100, batch_wait_timeout_s=10000)
    async def classify(self, inputs: List[Request]):
        # Run inference
        print("Our input array has length:", len(inputs))
        input_jsons = []
        for i in inputs:
            input_jsons.append(await i.json())
        return self.model([i['text'] for i in input_jsons])

    async def __call__(self, request: Request) -> dict:
        return await self.classify(request)
    
queryclassifier = QueryClassifier.bind()
# serve run server_queryclassifyv3minibatch:queryclassifier
# curl -X POST -d '{"text":"Hello world!"}' http://localhost:8000