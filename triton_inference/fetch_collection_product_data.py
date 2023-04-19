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
client_grpc = QdrantClient(url="http://localhost:6334", prefer_grpc=True)
DB_NAME = 'product_collection_clip_image'
DIM = 512
# %%
import s3fs
import os 
os.system('rm -rf tmp')
os.system('mkdir -p tmp')
fs = s3fs.S3FileSystem()
fs.get(
   'structured-data-dev/vector-db-multitask-ml/product-collection/collection_products_withclip_multitask_041223.parquet', 
   'tmp/payloads.parquet', recursive=False)
fs.get(
   'structured-data-dev/vector-db-multitask-ml/product-collection/collection_products_withclip_multitask_041223.npy', 
   'tmp/embs.npy', recursive=False)
# %%
from docarray import DocumentArray, Document
df_payloads = pd.read_parquet('tmp/payloads.parquet')
del df_payloads['img_embedding']
embs = np.load('tmp/embs.npy', mmap_mode='r')
assert embs.shape[1] == DIM and len(embs) == len(df_payloads)
os.system('rm -rf tmp')
print('len(df_payloads): ', len(df_payloads))
print('embs.shape: ', embs.shape)
print(f'nan status: {df_payloads.isna().any()}, auto fillna')
df_payloads.fillna(method='backfill', inplace=True)
assert not df_payloads.isna().any().any()
assert len(df_payloads) == len(set(df_payloads['product_id']))

df_payloads = df_payloads[['product_id']]
data = DocumentArray(
    [
        Document(id=i['product_id'], tags=i)
         for i in df_payloads.to_dict('records')
    ]
)
data.embeddings = embs

# %%
upload_data = True
if len(client.get_collections().collections) == 0 or \
      DB_NAME not in [i.name for i in client.get_collections().collections]:
   client.recreate_collection(
      collection_name=DB_NAME,
      vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
      hnsw_config=OptimizersConfigDiff(
         indexing_threshold=int(len(embs) - 1e3),
      )
   )
   print(f'create {DB_NAME} collection')
else:
   print(f'{DB_NAME} exist')
   upload_data = False

#%%
# c = 0
if upload_data:
   print('upload data')
   start_id = 0
   for batch in tqdm(chunks(data, 1000), desc="Upload data..."):
      # c += 1
      # if c == 10:
      #    break
      vectors = DocumentArray(batch).embeddings
      payloads = [i.tags for i in batch]
      ids = list(range(start_id, start_id + len(batch)))

      # client.upsert(
      #    collection_name=DB_NAME,
      #    points=Batch(
      #       ids=ids,
      #       payloads=payloads,
      #       vectors=vectors.tolist()
      #    )
      # )
      client_grpc.upload_collection(
         collection_name=DB_NAME,
         vectors=vectors,
         payload=payloads,
         ids=ids
      )

      start_id += len(batch)
else:
   print('skip upload')

#%%
print(f"{DB_NAME}: {client.get_collection(collection_name=DB_NAME)}")
os.system('mkdir -p model_repository/product_collection_keyword2annresponse_ensemble/1')