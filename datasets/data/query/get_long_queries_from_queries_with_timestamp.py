import pandas as pd
import json
from tqdm import tqdm

with open('wish_queries_with_timestamp_3wordsormore.json', 'w') as f:
    for df in tqdm(pd.read_json('wish_queries_with_timestamp.json', lines=True, chunksize=10000)):
        for i in df[df['query'].str.count(" ") >= 2].to_dict('records'):
            f.write(json.dumps(i) + '\n')