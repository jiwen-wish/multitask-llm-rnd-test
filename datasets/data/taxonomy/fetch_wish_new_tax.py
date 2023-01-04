# %%
from tahoe import create_external_table, drop_external_table, execute_async
from s3 import result_bucket, get_s3_file_keys, get_df_from_parquet
import pandas as pd
import logging
import dvc.api
import os

params = dvc.api.params_show()

tax_version = params['infer_wish_queries_new_taxonomy']['tax_version']
output_jsonl = os.path.join(os.path.dirname(__file__), 'wish_newtax.json')

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)

logging.info(
    (F"Fetch Wish new taxonomy {tax_version}"
        F" and save to {output_jsonl}")
)

# %%
# Define the export table
export_table = {
    "name": "newtax_export_tmp",
    "columns": [
        {"name": "category_tree_version", "type": "STRING"},
        {"name": "id", "type": "INTEGER"},
        {"name": "category_path", "type": "STRING"},
        {"name": "is_leaf", "type": "BOOLEAN"}
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
def insert_top_queries():
    """Insert top count user queries into table, along with gmv info"""
    q = f"""
    INSERT INTO {db}.{export_table['name']}
    SELECT category_tree_version, id, category_path, is_leaf
        from structured_data.wish_product_categories
    """
    execute_async(q)

# %%
# insert_top_queries(week_ago_dt)
insert_top_queries()

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

assert set(df_data['category_tree_version']) == set([tax_version])
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



