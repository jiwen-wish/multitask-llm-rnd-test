import os
from typing import Dict, List

import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    label_map: List

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        # more variables in https://github.com/triton-inference-server/python_backend/blob/main/src/python.cc
        path: str = os.path.join(args["model_repository"], args["model_version"], 'labels.txt')
        self.label_map = []
        with open(path, 'r') as f:
            for i in f:
                i = i.replace('\n', '')
                if len(i) > 0:
                    self.label_map.append(i)
        assert len(self.label_map) == 6038

    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        """
        Parse and tokenize each request
        :param requests: 1 or more requests received by Triton server.
        :return: text as input tensors
        """
        responses = []
        # for loop for batch requests (disabled in our case)
        for request in requests:
            # binary data typed back to string
            logits = pb_utils.get_input_tensor_by_name(request, "logits").as_numpy()
            print('logits: ', logits)
            
            # communicate the tokenization results to Triton server
            outputs = list()
            
            for input_name in ["categories", "weights"]:
                tensor_input = pb_utils.Tensor(input_name, "dummy")
                outputs.append(tensor_input)

            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)

        return responses