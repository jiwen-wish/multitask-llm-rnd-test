import asyncio
import aiohttp
import logging
import os 
import json
import pandas as pd
from tqdm import tqdm

OPENAI_KEY = os.environ['WISH_OPENAI_KEY_DEV']
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_KEY}"
}

async def call_oai(session, data):
    try:
        async with session.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=data
        ) as response:
            res = await response.json()
        return res
    except Exception as e:
        logging.error(f"{data} failed due to {e}")
        return None

async def call_oais(datas):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for data in datas:
            task = asyncio.ensure_future(call_oai(session, data))
            tasks.append(task)
        reses = await asyncio.gather(*tasks)
        return reses

batch_size = 100


with open('product_attribute_extraction_2023q2tli_041723_validprompt_041823_all_041823_oaiinfer_041823.json', 'w') as f:
    for chunk in tqdm(pd.read_json('product_attribute_extraction_2023q2tli_041723_validprompt_041823.json', 
            lines=True, chunksize=batch_size), total=int(1007384/batch_size)):
        res = asyncio.run(call_oais(chunk.prompt.tolist()))
        chunk['oai_response'] = res
        for i in chunk.to_dict('records'):
            f.write(json.dumps(i) + '\n')