# %%
from tahoe import create_external_table, drop_external_table, execute_async
from s3 import result_bucket, get_s3_file_keys, get_df_from_parquet
import pandas as pd
import logging
import dvc.api
import os

params = dvc.api.params_show()

timestamp_start = params['download_wish_queries']['start']
timestamp_end = params['download_wish_queries']['end']
output_jsonl = os.path.join(os.path.dirname(__file__), 'top_used_wish_queries.json')

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)

logging.info(
    (F"Fetch top used Wish queries from {timestamp_start} to {timestamp_end}"
        F" and save to {output_jsonl}")
)

# %%
# Define the export table
export_table = {
    "name": "top_query_export_tmp",
    "columns": [
        {"name": "query", "type": "STRING"},
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
def insert_top_queries(timestamp_start, timestamp_end):
    """Insert top count user queries into table, along with gmv info"""
    q = f"""
    INSERT INTO {db}.{export_table['name']}
    SELECT query, count(DISTINCT uid) as cnt, sum(gmv) as gmv
        from search.search_queries 
        where dt >='{timestamp_start}' and dt <= '{timestamp_end}'
        group by 1
        having count(DISTINCT uid)>100
    """
    execute_async(q)

# %%
# insert_top_queries(week_ago_dt)
insert_top_queries(timestamp_start, timestamp_end)

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
for file_key, file_size in file_keys:
    df_chunk = get_df_from_parquet(s3_bucket=result_bucket, s3_key=file_key)
    dfs.append(df_chunk)
    
df_data = pd.concat(dfs).sort_values('cnt', ascending=False)
df_data.head()

# %%
df_data.to_json(output_jsonl, orient="records", lines=True)

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
assert len(file_keys) == 0, "temp {} not deleted".format("sweeper_dev/{}".format(export_table["name"]))

# %%



