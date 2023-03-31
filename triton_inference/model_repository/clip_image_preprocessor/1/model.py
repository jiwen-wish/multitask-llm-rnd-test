import requests
import os
from typing import Dict, List

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import CLIPImageProcessor, TensorType
from concurrent.futures import ThreadPoolExecutor

import asyncio
import aiohttp
from io import BytesIO
from PIL import Image

BLANK_IMAGE_PATH = os.path.join(os.path.dirname(__file__), 'Black.png')

async def download_image(session, url):
    try:
        async with session.get(url) as response:
            image_bytes = await response.read()
        # Preprocess image here
        pil_image = Image.open(BytesIO(image_bytes))
        return pil_image
    except Exception as e:
        pb_utils.Logger.log_warn(f"Error downloading image at {url}: {str(e)}, use blank image")
        pil_image = Image.open(BLANK_IMAGE_PATH)
        return pil_image

async def download_images(urls):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            task = asyncio.ensure_future(download_image(session, url))
            tasks.append(task)
        images = await asyncio.gather(*tasks)
        return images

class TritonPythonModel:
    processor: CLIPImageProcessor

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        # more variables in https://github.com/triton-inference-server/python_backend/blob/main/src/python.cc
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        """
        Parse and tokenize each request
        :param requests: 1 or more requests received by Triton server.
        :return: text as input tensors
        """
        responses = []
        # for loop for batch requests (disabled in our case)
        urls = []
        chunk_sizes = []
        for request in requests:
            # binary data typed back to string
            url = [
                t[0].decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "image_url")
                .as_numpy()
                .tolist()
            ]
            urls += url
            chunk_sizes.append(len(url))
        urls = ["https://canary.contestimg.wish.com/api/webimage/61b241a3a4ee2ecaf2f63c77-large.jpg?cache_buster=bbeee1fdb460a1d12bc266824914e030"] * len(urls)

        images = asyncio.run(download_images(urls))

        inputs = self.processor(images=images, return_tensors=TensorType.NUMPY)

        responses = []
        rsum = 0
        for ind in range(len(requests)):
            outputs = list()
            for input_name in ["pixel_values"]:
                tensor_input = pb_utils.Tensor(input_name, inputs[input_name][rsum:rsum+chunk_sizes[ind]])
                outputs.append(tensor_input)
            rsum += chunk_sizes[ind]
            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)

        return responses