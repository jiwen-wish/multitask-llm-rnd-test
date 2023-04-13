import numpy as np
import triton_python_backend_utils as pb_utils
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, OptimizersConfigDiff
from tqdm import tqdm
from qdrant_client.http.models import SearchRequest
import json 
import asyncio
import aiohttp
import json
import os

class TritonPythonModel:

    def initialize(self, args):
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        self.oai_key = os.environ['OPENAI_API_KEY']

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.oai_key}"
        }

    async def call_oai(self, session, url, data):
        try:
            async with session.post(
                url,
                headers=self.headers,
                data=data
            ) as response:
                res = await response.json()
            return res, True
        except Exception as e:
            pb_utils.Logger.log_warn(f"Calling openai {url} with {data} failed due to {e}")
            return {}, False

    async def call_oais(self, urls, datas):
        async with aiohttp.ClientSession() as session:
            tasks = []
            for url, data in zip(urls, datas):
                task = asyncio.ensure_future(self.call_oai(session, url, data))
                tasks.append(task)
            reses = await asyncio.gather(*tasks)
            return reses

    def execute(self, requests):
        all_urls = []
        all_datas = []
        chunk_sizes = []
        # for loop for batch requests (disabled in our case)
        for request in requests:
            # binary data typed back to string
            urls = [
                t[0].decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "urls")
                .as_numpy()
                .tolist()
            ]

            all_urls += urls

            data = [
                t[0].decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "requests")
                .as_numpy()
                .tolist()
            ]

            all_datas += data
            
            chunk_sizes.append(len(urls))
            
        all_reses_success = asyncio.run(self.call_oais(urls, all_datas))
        all_reses = np.array([json.dumps(i[0]) for i in all_reses_success])
        all_success = np.array(all_reses_success)

        inputs = {}
        inputs["responses"] = all_reses.astype(np.dtype('S')).reshape(-1,1)
        inputs['success'] = all_success.astype(bool).reshape(-1,1)

        responses = []
        rsum = 0
        for ind in range(len(requests)):
            outputs = list()
            for input_name in ["responses", "success"]:
                tensor_input = pb_utils.Tensor(input_name, inputs[input_name][rsum:rsum+chunk_sizes[ind]])
                outputs.append(tensor_input)
            rsum += chunk_sizes[ind]
            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)
            
        return responses