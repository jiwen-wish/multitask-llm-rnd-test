#%%
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Batch, OptimizersConfigDiff
from tqdm import tqdm
import pandas as pd
import numpy as np 

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

client = QdrantClient(url="http://localhost:6333")
DB_NAME = 'product_collection_clip_image'
# %%
import s3fs
import os 
os.system('rm -rf tmp')
os.system('mkdir -p tmp')
fs = s3fs.S3FileSystem()
fs.get(
   'structured-data-dev/vector-db-multitask-ml/product-collection/collection_products_withclip_peterhull_top10000_041223.bin', 
   'tmp/data.bin', recursive=True)
# %%
from docarray import DocumentArray
data = DocumentArray.load('tmp/data.bin')
os.system('rm -rf tmp')
print('len(data): ', len(data))
# %%
upload_data = True
if len(client.get_collections().collections) == 0 or \
      DB_NAME not in [i.name for i in client.get_collections().collections]:
   client.recreate_collection(
      collection_name=DB_NAME,
      vectors_config=VectorParams(size=512, distance=Distance.COSINE),
      hnsw_config=OptimizersConfigDiff(
         indexing_threshold=int(len(data) - 1e3),
      )
   )
   print(f'create {DB_NAME} collection')
else:
   print(f'{DB_NAME} exist')
   upload_data = False

#%%
if upload_data:
   print('upload data')
   start_id = 0
   for batch in tqdm(chunks(data, 1000), desc="Upload data..."):
      vectors = DocumentArray(batch).embeddings[:,:512]
      payloads = [i.tags for i in batch]
      tmp = pd.DataFrame(payloads)
      tmp['rating_count'] = tmp['rating_count'].fillna(0.0)
      payloads = tmp.to_dict('records')
      ids = list(range(start_id, start_id + len(batch)))

      client.upsert(
         collection_name=DB_NAME,
         points=Batch(
            ids=ids,
            payloads=payloads,
            vectors=vectors.tolist()
         )
      )

      start_id += len(batch)
else:
   print('skip upload')

#%%
print(f"{DB_NAME}: {client.get_collection(collection_name=DB_NAME)}")
os.system('mkdir -p model_repository/product_collection_keyword2annresponse_ensemble/1')