import numpy as np
import triton_python_backend_utils as pb_utils
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, OptimizersConfigDiff
from tqdm import tqdm
from qdrant_client.http.models import SearchRequest
import json 

class TritonPythonModel:

    def initialize(self, args):
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        self.ann_client = QdrantClient(url="http://localhost:6333")

    def execute(self, requests):
        all_embs = []
        all_collections = []
        all_limits = []
        chunk_sizes = []
        # for loop for batch requests (disabled in our case)
        for request in requests:
            # binary data typed back to string
            
            all_embs.append(pb_utils.get_input_tensor_by_name(request, "embs").as_numpy())

            collection = [
                t[0].decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "collections")
                .as_numpy()
                .tolist()
            ]

            all_collections += collection
            all_limits.append(pb_utils.get_input_tensor_by_name(request, "limits").as_numpy())
            chunk_sizes.append(len(collection))
            
        all_embs = np.vstack(all_embs)
        all_limits = np.vstack(all_limits)
        all_collections = np.array(all_collections)
        all_reses = np.array([None] * len(all_embs))
        all_success = np.array([True] * len(all_embs))

        for collection in set(all_collections):
            indicator = (all_collections == collection)
            sub_embs = all_embs[indicator]
            sub_limits = all_limits[indicator]
            try:
                search_queries = [
                    SearchRequest(
                        vector=sub_embs[ind].tolist(),
                        limit=sub_limits[ind], 
                        with_payload=True   
                    ) for ind in range(len(sub_limits))
                ]

                res = self.ann_client.search_batch(
                    collection_name=collection,
                    requests=search_queries
                )
                
                all_reses[indicator] = [json.dumps([j.dict() for j in i]) for i in res]
            except Exception as e:
                pb_utils.Logger.log_warn(f"ann search for {collection} failed due to {e}")
                all_success[indicator] = False
                all_reses[indicator] = [json.dumps([])] * len(sub_embs)
        assert (all_reses != None).all()
        

        inputs = {}
        inputs["responses"] = all_reses.astype(np.dtype('S')).reshape(-1,1)
        inputs['success'] = all_success.astype(bool).reshape(-1,1)

        responses = []
        rsum = 0
        for ind in range(len(requests)):
            outputs = list()
            for input_name in ["responses", "success"]:
                tensor_input = pb_utils.Tensor(input_name, inputs[input_name][rsum:rsum+chunk_sizes[ind]])
                outputs.append(tensor_input)
            rsum += chunk_sizes[ind]
            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)
            
        return responses