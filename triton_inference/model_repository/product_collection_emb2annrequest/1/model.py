import numpy as np
import json
import triton_python_backend_utils as pb_utils

class TritonPythonModel:

    def initialize(self, args):
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        self.collection_name = 'product_collection_clip_image'
        self.limit_count = 1000
        model_config = json.loads(args['model_config'])
        
        embs_out_config = pb_utils.get_output_config_by_name(
            model_config, "embs")
        self.embs_out_dtype = pb_utils.triton_string_to_numpy(
            embs_out_config['data_type'])
        
        collections_out_config = pb_utils.get_output_config_by_name(
            model_config, "collections")
        self.collections_out_dtype = pb_utils.triton_string_to_numpy(
            collections_out_config['data_type'])
        
        limits_out_config = pb_utils.get_output_config_by_name(
            model_config, "limits")
        self.limits_out_dtype = pb_utils.triton_string_to_numpy(
            limits_out_config['data_type'])

    def execute(self, requests):
        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            embs = pb_utils.get_input_tensor_by_name(request, "embs").as_numpy()
            embs_out = pb_utils.Tensor("embs", embs.astype(self.embs_out_dtype))
            collections_out = pb_utils.Tensor("collections", 
                np.array([self.collection_name] * len(embs)).astype(
                    self.collections_out_dtype).reshape(-1,1))
            limits_out = pb_utils.Tensor("limits", 
                np.array([self.limit_count] * len(embs)).astype(
                    self.limits_out_dtype).reshape(-1,1))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[embs_out, collections_out, limits_out])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses
