# %%
from tahoe import create_external_table, drop_external_table, execute_async
from s3 import result_bucket, get_s3_file_keys, get_df_from_parquet
import pandas as pd
import logging
import dvc.api
import os
from tqdm import tqdm

params = dvc.api.params_show()

output_jsonl = os.path.join(os.path.dirname(__file__), 'wish_queries_inferred_newtax_multiple.json')

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)

logging.info(
    (F"Fetch Wish queries with inferred new taxonomy for multiple tables"
        F" and save to {output_jsonl}")
)

# %%
# Define the export table
export_table = {
    "name": "query_inferred_export_tmp",
    "columns": [
        {"name": "query", "type": "STRING"},
        {"name": "norm", "type": "FLOAT"},
        {"name": "categories", "type": "STRING"},
        {"name": "category_names", "type": "STRING"},
        {"name": "weights", "type": "STRING"},
        {"name": "search_table_name", "type": "STRING"}
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

#%%
def get_multiple_tables():
    table_res = execute_async(
        query=f"""
        SELECT DISTINCT (table_name) FROM information_schema.tables 
        WHERE table_schema = 'search'
        AND table_name like 'query_new_category_inference_%'
        """
    )
    table_names = [i[0] for i in table_res]
    return table_names

# %%
def insert_queries(table_name):
    """Insert top count user queries into table, along with gmv info"""
    q = f"""
    INSERT INTO {db}.{export_table['name']}
    SELECT query, norm, categories, category_names, weights, '{table_name}' as search_table_name
        from search.{table_name}
    """
    execute_async(q)
    

# %%
# insert_top_queries(week_ago_dt)
for table_name in tqdm(get_multiple_tables()):
    insert_queries(table_name)

# %%
# Check again the number of rows in the external table
q = f"SELECT COUNT(*) FROM {db}.{export_table['name']}"
print(execute_async(q))

# %%
# Show the data files for the external table
file_keys = get_s3_file_keys(s3_bucket=result_bucket, 
    s3_prefix="sweeper_dev/{}".format(export_table["name"]))

# %%
# Import the parquet file as data frame
# Alternatively, you can use boto3 to directly download the file to disk
# For hundreds of data files, download using multiprocessing

dfs = []
for file_key, file_size in file_keys:
    df_chunk = get_df_from_parquet(s3_bucket=result_bucket, s3_key=file_key)
    dfs.append(df_chunk)
    
df_data = pd.concat(dfs)

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



