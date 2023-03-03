from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from transformers import pipeline
import asyncio
from dvc.api import DVCFileSystem
import tempfile
import sys
import os
import importlib

repo = 'git@github.com:ContextLogic/multitask-llm-rnd.git'
path = '/modelling/notebooks/convert_pl_to_hf_ckpt/query_classify_v3_mini/hf_ckpt'
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

async def homepage(request):
    payload = await request.body()
    string = payload.decode("utf-8")
    response_q = asyncio.Queue()
    await request.app.model_queue.put((string, response_q))
    output = await response_q.get()
    return JSONResponse(output)


async def server_loop(q):
    with tempfile.TemporaryDirectory() as tmpdirname:
        fs = DVCFileSystem(repo, subrepos=True)
        fs.get(path, tmpdirname, recursive=True)
        pipe = import_path(f"{tmpdirname}/load_model.py").load_pipeline()
    while True:
        strings = []
        queues = []
        (string, rq) = await q.get()
        strings.append(string)
        queues.append(rq)
        while not q.empty() and len(strings) < max_batch_size:
            (string, rq) = await q.get()
            strings.append(string)
            queues.append(rq)

        outs = pipe(strings, batch_size=len(strings))
        for (rq, out) in zip(queues, outs):
            await rq.put(out)


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