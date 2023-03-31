import grequests
import requests
import os
from typing import Dict, List

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoProcessor, TensorType
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from io import BytesIO

class TritonPythonModel:
    processor: AutoProcessor

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        # more variables in https://github.com/triton-inference-server/python_backend/blob/main/src/python.cc
        path: str = os.path.join(args["model_repository"], args["model_version"])
        self.blank_path = os.path.join(path, "Black.png")
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def download_image(self, url):
        try:
            return requests.get(url, stream=True).raw
            # return Image.open(requests.get(url, stream=True).raw)
        except Exception as e:
            pb_utils.Logger.log_warn(f"Download image from {url} failed due to {e}, will use blank image at {self.blank_path}")
            return Image.open(self.blank_path)

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
        urls = ["http://images.cocodataset.org/val2017/000000039769.jpg"] * len(urls)

        with ThreadPoolExecutor(max_workers=min(4, len(urls))) as executor:
            images = list(executor.map(self.download_image, urls))

        # inputs = self.processor(images=images, return_tensors=TensorType.NUMPY)
        inputs = {"pixel_values": np.random.random((len(requests), 3, 244, 244))}
        # communicate the tokenization results to Triton server
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