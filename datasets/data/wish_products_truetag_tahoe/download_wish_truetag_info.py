# %%
from tahoe import create_external_table, drop_external_table, execute_async
from s3 import result_bucket, get_s3_file_keys, get_df_from_parquet
import pandas as pd
import logging
import os
from tqdm import tqdm
import gzip
import json

output_json = os.path.join(os.path.dirname(__file__), 'true_tag_info.json')

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)

logging.info(
    F"Fetch Wish Tahoe table [search.true_tag_info]"
)

# %%
# Define the export table
export_table = {
    "name": "truetag_tmp",
    "columns": [
        {"name": "id", "type": "STRING"},
        {"name": "true_tag_ids", "type": "ARRAY<STRING>"},
        {"name": "true_tag_names", "type": "ARRAY<STRING>"}
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
def insert_truetag_queries():
    """Insert top count user queries into table, along with gmv info"""
    q = f"""
    INSERT INTO {db}.{export_table['name']}
    SELECT id, true_tag_ids, true_tag_names
    from search.true_tag_info
    """
    execute_async(q)

# %%
insert_truetag_queries()


# %%
# Show the data files for the external table
file_keys = get_s3_file_keys(s3_bucket=result_bucket, 
    s3_prefix="sweeper_dev/{}".format(export_table["name"]))


# %%
# Import the parquet file as data frame
# Alternatively, you can use boto3 to directly download the file to disk
# For hundreds of data files, download using multiprocessing
logging.info(f'Process data')

truetags = {}

for file_key, file_size in tqdm(file_keys):
    df_chunk = get_df_from_parquet(s3_bucket=result_bucket, s3_key=file_key)
    dummy = 2
    for dat in df_chunk.to_dict('records'):
        truetags[dat['id']] = [i.strip().lower() for i in dat['true_tag_names'][::-1]]

recs = []
for i in truetags:
    recs.append({'true_tag_ids': i, 'category': truetags[i]})

df_out = pd.DataFrame(recs)
df_out.to_json(output_json, orient='records', lines=True)


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
