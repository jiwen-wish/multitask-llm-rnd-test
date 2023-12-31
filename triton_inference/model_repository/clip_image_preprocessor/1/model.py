#%%
import os
from typing import Dict, List

import numpy as np
import triton_python_backend_utils as pb_utils
import asyncio
import aiohttp
from io import BytesIO
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import torch

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def preprocess(n_px=224):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

BLANK_IMAGE_PATH = os.path.join(os.path.dirname(__file__), 'Black.png')

#%%
class TritonPythonModel:

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        # more variables in https://github.com/triton-inference-server/python_backend/blob/main/src/python.cc
        self.path: str = os.path.join(args["model_repository"], args["model_version"])
        self.processor = preprocess()

    async def download_image(self, session, url):
        try:
            async with session.get(url) as response:
                image_bytes = await response.read()
            # Preprocess image here
            pil_image = Image.open(BytesIO(image_bytes))
            return pil_image, True
        except Exception as e:
            pb_utils.Logger.log_warn(f"Model: {self.path} - Error downloading image at {url}: {str(e)}, use blank image")
            pil_image = Image.open(BLANK_IMAGE_PATH)
            return pil_image, False

    async def download_images(self, urls):
        async with aiohttp.ClientSession() as session:
            tasks = []
            for url in urls:
                task = asyncio.ensure_future(self.download_image(session, url))
                tasks.append(task)
            images_successes = await asyncio.gather(*tasks)
            return images_successes

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
        # urls = ["https://canary.contestimg.wish.com/api/webimage/61b241a3a4ee2ecaf2f63c77-large.jpg?cache_buster=bbeee1fdb460a1d12bc266824914e030"] * len(urls)

        images_successes = asyncio.run(self.download_images(urls))
        images, successes = list(zip(*images_successes))
        
        inputs = {}
        inputs["pixel_values"] = torch.cat([self.processor(i).unsqueeze(0) for i in images]).numpy()
        inputs['image_download_success'] = np.array(successes, dtype=bool).reshape(len(images), 1)

        responses = []
        rsum = 0
        for ind in range(len(requests)):
            outputs = list()
            for input_name in ["pixel_values", "image_download_success"]:
                tensor_input = pb_utils.Tensor(input_name, inputs[input_name][rsum:rsum+chunk_sizes[ind]])
                outputs.append(tensor_input)
            rsum += chunk_sizes[ind]
            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)

        return responses