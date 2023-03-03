from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from transformers import pipeline
import asyncio

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import os
import tempfile
import importlib
import sys
from dvc.api import DVCFileSystem

repo = 'git@github.com:ContextLogic/multitask-llm-rnd.git'
path = '/modelling/notebooks/convert_pl_to_hf_ckpt/query_classify_v3_mini/hf_ckpt'

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


def load_pipeline():
    """
    pipe(['Classify query: apple'], batch_size=10)
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        fs = DVCFileSystem(repo, subrepos=True)
        fs.get(path, tmpdirname, recursive=True)
        pipe = import_path(f"{tmpdirname}/load_model.py").load_pipeline()
    return pipe

async def homepage(request):
    payload = await request.body()
    string = payload.decode("utf-8")
    response_q = asyncio.Queue()
    await request.app.model_queue.put((string, response_q))
    output = await response_q.get()
    return JSONResponse(output)


async def server_loop(q):
    pipe = load_pipeline()

    while True:
        strings = []
        queues = []
        (string, rq) = await q.get()
        strings.append(string)
        queues.append(rq)

        take = 4
        while not q.empty() and take > 0:
            (string, rq) = await q.get()
            strings.append(string)
            queues.append(rq)

        if len(strings) == 0:
            continue

        outs = pipe(strings, batch_size=len(strings))
        for (rq, out) in zip(queues, outs):
            await rq.put(out)

        strings = []
        queues = []

app = Starlette(
    routes=[
        Route("/", homepage, methods=["POST"]),
    ],
)


@app.on_event("startup")
async def startup_event():
    q = asyncio.Queue()
    app.model_queue = q
    asyncio.create_task(server_loop(q))