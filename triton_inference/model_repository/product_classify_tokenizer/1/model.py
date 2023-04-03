import os
from typing import Dict, List

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, PreTrainedTokenizer, TensorType

class TritonPythonModel:
    tokenizer: PreTrainedTokenizer

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        # more variables in https://github.com/triton-inference-server/python_backend/blob/main/src/python.cc
        
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', fast=True)

    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        """
        Parse and tokenize each request
        :param requests: 1 or more requests received by Triton server.
        :return: text as input tensors
        """
        responses = []
        # for loop for batch requests (disabled in our case)
        queries = []
        chunk_sizes = []
        for request in requests:
            # binary data typed back to string
            query = [
                t[0].decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "text")
                .as_numpy()
                .tolist()
            ]
            queries += query
            chunk_sizes.append(len(query))

        tokens: Dict[str, np.ndarray] = self.tokenizer(
            text=queries, return_tensors=TensorType.NUMPY, 
            max_length=512, truncation=True, padding='max_length'
        )

        # tensorrt uses int32 as input type, ort uses int64
        tokens = {k: v.astype(np.int32) for k, v in tokens.items()}
        # communicate the tokenization results to Triton server
        rsum = 0
        for ind in range(len(requests)):
            outputs = list()
            for input_name in ["input_ids", "attention_mask"]:
                tensor_input = pb_utils.Tensor(input_name, tokens[input_name][rsum:rsum+chunk_sizes[ind]])
                outputs.append(tensor_input)
            rsum += chunk_sizes[ind]
            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)

        return responses