import os
from typing import Dict, List

import numpy as np
import triton_python_backend_utils as pb_utils
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

sigmoid_vec = np.vectorize(sigmoid)

class TritonPythonModel:
    label_map: np.array

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        # more variables in https://github.com/triton-inference-server/python_backend/blob/main/src/python.cc
        path: str = os.path.join(args["model_repository"], args["model_version"], 'labels.txt')
        label_map = []
        with open(path, 'r') as f:
            for i in f:
                i = i.replace('\n', '')
                if len(i) > 0:
                    label_map.append(int(i))
        assert len(label_map) == 6038
        self.label_map = np.array([label_map], dtype=np.dtype('int32'))

    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        """
        Parse and tokenize each request
        :param requests: 1 or more requests received by Triton server.
        :return: text as input tensors
        """
        responses = []
        all_logits = []
        chunk_sizes = []
        # for loop for batch requests (disabled in our case)
        for request in requests:
            # binary data typed back to string
            i = pb_utils.get_input_tensor_by_name(request, "logits").as_numpy()
            all_logits.append(i)
            chunk_sizes.append(len(i))
            
        logits = np.vstack(all_logits)
        top_10_inds = np.argsort(-logits, axis=1)[:, :10]
        top_10_cats = np.take_along_axis(self.label_map, top_10_inds, axis=1)
        top_10_probs = sigmoid_vec(np.take_along_axis(logits, top_10_inds, axis=1))
        top_10_cats_filter_unk = []
        top_10_probs_filter_unk = []
        for c, p in zip(top_10_cats, top_10_probs):
            cs = [] 
            ps = []
            for c_, p_ in zip(c, p):
                if c_ == -1:
                    break 
                else:
                    cs.append(str(c_))
                    ps.append(str(p_))
            top_10_cats_filter_unk.append((",".join(cs)).encode('utf-8') )
            top_10_probs_filter_unk.append((",".join(ps)).encode('utf-8') )
        
        rsum = 0
        for ind in range(len(requests)):
            outputs = [ 
                pb_utils.Tensor('categories', 
                    np.array(top_10_cats_filter_unk[rsum:chunk_sizes[ind]], 
                        dtype=np.dtype('S')).reshape(-1,1)),
                pb_utils.Tensor('weights', 
                    np.array(top_10_probs_filter_unk[rsum:chunk_sizes[ind]], 
                        dtype=np.dtype('S')).reshape(-1,1))
            ]
            rsum += chunk_sizes[ind]

            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)

        return responses