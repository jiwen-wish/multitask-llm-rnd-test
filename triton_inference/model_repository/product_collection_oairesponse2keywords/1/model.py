import numpy as np
import json
import triton_python_backend_utils as pb_utils
import asyncio
import pandas as pd

class TritonPythonModel:

    def initialize(self, args):
        model_config = json.loads(args['model_config'])
        self.limit_count = 50
        texts_out_config = pb_utils.get_output_config_by_name(
            model_config, "texts")
        self.texts_out_dtype = pb_utils.triton_string_to_numpy(
            texts_out_config['data_type'])

    def extract_data_from_oaitext(self, data):
        # Split the data into lines and remove any empty lines
        lines = [line.strip() for line in data.split("\n") if line.strip()]

        # Create a list of tuples containing the Collection and Interest data
        data_list = []
        for line in lines:
            interest = line.split("\t")[0]
            # should look like "<num>. <keywords>"
            data_list.append(interest.split(" ", 1)[1])
        return data_list

    def execute(self, requests):
        responses_out = []
        for request in requests:
            # Get INPUT0
            responses = [
                t[0].decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "responses")
                .as_numpy()
                .tolist()
            ]
            success = pb_utils.get_input_tensor_by_name(request, "responses"
                                                        ).as_numpy().reshape(-1).tolist()

            texts = [] 
            for res, suc in zip(responses, success):
                text = []
                if suc:
                    try:
                        res_text = json.loads(res)["choices"][0]["text"]
                        res_tags = self.extract_data_from_oaitext(res_text)
                        text += list(set([i for i in res_tags if len(i) > 0]))[:50]
                    except Exception as e:
                        pb_utils.Logger.log_warn(f"parsing oai response {res} failed due to {e}, padding results with empty strings")
                else:
                    pb_utils.Logger.log_warn(f"oai response {res} failed, padding results with empty strings")
                text += [""] * (self.limit_count - len(text))
                texts.append(text)

            texts_out = pb_utils.Tensor("texts", 
                np.array(texts).astype(
                    self.texts_out_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[texts_out])
            responses_out.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses_out