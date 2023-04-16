#%%
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Batch, OptimizersConfigDiff
from tqdm import tqdm
from qdrant_client.http.api_client import AsyncApis
from qdrant_client.http.models import SearchRequest
import numpy as np 
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

client = QdrantClient(url="http://localhost:6333")
client_grpc = QdrantClient(url="http://localhost:6334", prefer_grpc=True)
aapi = AsyncApis('http://localhost:6333')
#%%
collection_name = "tmp"
limit = 1000
batch_size = 10

# client.recreate_collection(
#       collection_name=collection_name,
#       vectors_config=VectorParams(size=512, distance=Distance.COSINE),
#       hnsw_config=OptimizersConfigDiff(
#          indexing_threshold=int(1e6 - 1e3),
#       )
#    )
# # %%
# start = 0
# for i in tqdm(range(1000)):
#     vectors = np.random.random((1000, 512))
#     client_grpc.upload_collection(
#         collection_name=collection_name,
#         vectors=vectors,
#         ids=list(range(start, start+1000))
#     )
#     start += 1000

#%%
print(client.get_collection(collection_name=collection_name))
#%%
async def search_async():
    reses = []
    for i in range(batch_size):
        query_vector = np.random.random(512)
        search_result = aapi.points_api.search_points(
            collection_name=collection_name,
            search_request=SearchRequest.construct(
                vector=query_vector.tolist(),
                limit=limit,
                with_vector=False,
                with_payload=False
            ),
        )
        reses.append(search_result)
    reses_ = await asyncio.gather(*reses)
    return reses_

tsart = time.time()
reses_ = asyncio.run(search_async())
tend = time.time()
print('async: ', tend - tsart)
# %%
tsart = time.time()
search_queries = [
    SearchRequest(
        vector=np.random.random(512).tolist(),
        limit=1000, 
        with_payload=False,
        with_vector=False 
    ) for ind in range(batch_size)
]

res = client.search_batch(
    collection_name=collection_name,
    requests=search_queries
)
tend = time.time()
print('batch: ', tend - tsart)

#%%
tsart = time.time()
with ThreadPoolExecutor(max_workers=batch_size) as executor:
    futures = [executor.submit(client.search_batch, 
        collection_name=collection_name, 
        requests=[SearchRequest(
            vector=np.random.random(512).tolist(),
            limit=1000, 
            with_payload=False,
            with_vector=False 
        )]) for _ in range(batch_size)]
    results = [f.result() for f in futures]
tend = time.time()
print('thread: ', tend - tsart)

# %%
tsart = time.time()
for i in range(batch_size):
    search_queries = [
        SearchRequest(
            vector=np.random.random(512).tolist(),
            limit=1000, 
            with_payload=False,
            with_vector=False 
        ) for ind in range(1)
    ]
    res = client.search_batch(
        collection_name=collection_name,
        requests=search_queries
    )
tend = time.time()
print('sequential: ', tend - tsart)
