import numpy as np
import json
import triton_python_backend_utils as pb_utils

class TritonPythonModel:

    def initialize(self, args):
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        self.oai_url = 'https://api.openai.com/v1/completions'
        self.oai_hparams = dict(
            model="text-davinci-003", 
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0 # 0.7
        )
        self.oai_prompt = "List up to 50 popular interests, themes, or product attributes to use as keywords within OpenAI's CLIP model for e-commerce products in the '{theme}' category."

        model_config = json.loads(args['model_config'])
        
        urls_out_config = pb_utils.get_output_config_by_name(
            model_config, "urls")
        self.urls_out_dtype = pb_utils.triton_string_to_numpy(
            urls_out_config['data_type'])
        
        requests_out_config = pb_utils.get_output_config_by_name(
            model_config, "requests")
        self.requests_out_dtype = pb_utils.triton_string_to_numpy(
            requests_out_config['data_type'])
        

    def execute(self, requests):
        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            themes = [
                t[0].decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "theme")
                .as_numpy()
                .tolist()
            ]
            urls_out = pb_utils.Tensor("urls", 
                np.array([self.oai_url] * len(themes)).astype(
                    self.urls_out_dtype).reshape(-1,1))

            requests_out = pb_utils.Tensor("requests", 
                np.array([
                    json.dumps({
                        "prompt": self.oai_prompt.format(theme=i),
                        **self.oai_hparams
                    }) for i in themes]).astype(
                    self.requests_out_dtype).reshape(-1,1))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[urls_out, requests_out])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses
