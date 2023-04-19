import numpy as np
import json
import triton_python_backend_utils as pb_utils
import asyncio
import pandas as pd

class TritonPythonModel:

    def initialize(self, args):
        self.inf_model_name = "product_collection_keyword2annresponse_ensemble"
        self.limit_count = 1000
        model_config = json.loads(args['model_config'])
        
        product_ids_out_config = pb_utils.get_output_config_by_name(
            model_config, "product_ids")
        self.product_ids_out_dtype = pb_utils.triton_string_to_numpy(
            product_ids_out_config['data_type'])
        
        scores_out_config = pb_utils.get_output_config_by_name(
            model_config, "scores")
        self.scores_out_dtype = pb_utils.triton_string_to_numpy(
            scores_out_config['data_type'])

    async def execute(self, requests):
        assert len(requests) == 1
        request = requests[0]
        texts = [
            t.decode("UTF-8")
            for t in pb_utils.get_input_tensor_by_name(request, "texts")
            .as_numpy()
            .tolist()
        ]
        inference_response_awaits = []
        for text in texts:
            # Create inference request object
            input_tensor = pb_utils.Tensor("text", np.array([text]).astype(np.dtype('S')).reshape(1,1))
            infer_request = pb_utils.InferenceRequest(
                model_name=self.inf_model_name,
                requested_output_names=["responses", "success"],
                inputs=[input_tensor])

            # Store the awaitable inside the array. We don't need
            # the inference response immediately so we do not `await`
            # here.
            inference_response_awaits.append(infer_request.async_exec())
        
        inference_responses = await asyncio.gather(
                *inference_response_awaits)
        for infer_response in inference_responses:
            # Make sure that the inference response doesn't have an error.
            # If it has an error and you can't proceed with your model
            # execution you can raise an exception.
            if infer_response.has_error():
                raise pb_utils.TritonModelException(
                    infer_response.error().message())
        
        all_scores = []
        all_pids = []
        for ind, i in enumerate(inference_responses):
            inf_success = pb_utils.get_output_tensor_by_name(i, "success")
            if inf_success.as_numpy()[0][0]:
                inf_responses = pb_utils.get_output_tensor_by_name(i, "responses")
                inf_responses_decoded = json.loads(inf_responses.as_numpy()[0][0])
                all_scores += [j['score'] for j in inf_responses_decoded]
                all_pids += [j['payload']['product_id'] for j in inf_responses_decoded]
            else:
                pb_utils.Logger.log_warn(f"ann search for keyword {texts[ind]} failed, skip")
        df = pd.DataFrame({"pid": all_pids, "score": all_scores})
        df2 = df.groupby('pid').mean().reset_index().sort_values(by='score', ascending=False).head(self.limit_count)
        
        responses = [
            pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("product_ids", df2['pid'].to_numpy().astype(self.product_ids_out_dtype).reshape(-1)),
                pb_utils.Tensor("scores", df2['score'].to_numpy().astype(self.scores_out_dtype).reshape(-1))
            ])
        ]
        
        return responses

        