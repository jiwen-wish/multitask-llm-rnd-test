# %%
from tahoe import create_external_table, drop_external_table, execute_async
from s3 import result_bucket, get_s3_file_keys, get_df_from_parquet
import pandas as pd
import logging
import dvc.api
import os
from tqdm import tqdm
import json

params = dvc.api.params_show()
timestamp_start = params['download_wish_queries_with_timestamp']['start']
timestamp_end = params['download_wish_queries_with_timestamp']['end']
output_jsonl = os.path.join(os.path.dirname(__file__), 'wish_queries_with_timestamp.json')

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)

logging.info(
    (F"Fetch top used Wish queries before {timestamp_end}"
        F" and save to {output_jsonl}")
)

# %%
# Define the export table
export_table = {
    "name": "query_export_tmp",
    "columns": [
        {"name": "query", "type": "STRING"},
        {"name": "min_timestamp", "type": "INTEGER"},
        {"name": "max_timestamp", "type": "INTEGER"},
        {"name": "min_dt", "type": "STRING"},
        {"name": "max_dt", "type": "STRING"},
        {"name": "cnt", "type": "INTEGER"},
        {"name": "gmv", "type": "FLOAT"},
    ]
}
db = "sweeper_dev"
# %%
# Optional: drop the external table
drop_external_table(
    db=db,
    table_name=export_table["name"],
    delete_files=True,
    s3_bucket=result_bucket,
    s3_prefix="sweeper_dev/{}".format(export_table["name"]), 
)

# %%
# Create the export table
create_external_table(
    table_name=export_table["name"],
    table_definition=export_table,
    db=db,
    bucket=result_bucket
)

# %%
# Check the number of rows in the external table
q = f"SELECT COUNT(*) FROM {db}.{export_table['name']}"
assert execute_async(q)[0] == (0,), f"{db}.{export_table['name']} not empty"

# %%
def insert_queries(timestamp_start, timestamp_end):
    """Insert queries"""
    q = f"""
    INSERT INTO {db}.{export_table['name']}
    SELECT query, MIN(timestamp) as min_timestamp, MAX(timestamp) as max_timestamp,
    MIN(dt) as min_dt, MAX(dt) as max_dt, COUNT(request_id) as cnt, SUM(gmv) as gmv
    FROM search.search_queries 
    WHERE dt >='{timestamp_start}' and dt <= '{timestamp_end}'
    GROUP BY query ORDER BY min_timestamp
    """
    execute_async(q)

# %%
# insert_top_queries(week_ago_dt)
insert_queries(timestamp_start, timestamp_end)

# %%
# Check again the number of rows in the external table
q = f"SELECT COUNT(*) FROM {db}.{export_table['name']}"
execute_async(q)

# %%
# Show the data files for the external table
file_keys = get_s3_file_keys(s3_bucket=result_bucket, 
    s3_prefix="sweeper_dev/{}".format(export_table["name"]))
file_keys

# %%
# Import the parquet file as data frame
# Alternatively, you can use boto3 to directly download the file to disk
# For hundreds of data files, download using multiprocessing

dfs = []
with open(output_jsonl, 'w') as f:
    for file_key, file_size in tqdm(file_keys):
        df_chunk = get_df_from_parquet(s3_bucket=result_bucket, s3_key=file_key)
        for i in df_chunk.to_dict('records'):
            try:
                f.write(json.dumps(i) + '\n')
            except Exception as e:
                print(i, e)

# %%
# Optional: drop the external table
drop_external_table(
    db=db,
    table_name=export_table["name"],
    delete_files=True,
    s3_bucket=result_bucket,
    s3_prefix="sweeper_dev/{}".format(export_table["name"]), 
)

# %%
# Show the data files for the external table
file_keys = get_s3_file_keys(s3_bucket=result_bucket, s3_prefix="sweeper_dev/{}".format(export_table["name"]))
try:
    assert len(file_keys) == 0, "temp {} not deleted".format("sweeper_dev/{}".format(export_table["name"]))
except Exception as e:
    print(e)

# %%



