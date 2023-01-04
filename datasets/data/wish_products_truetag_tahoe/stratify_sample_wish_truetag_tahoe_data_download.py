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

df_truetag_mapping = pd.read_json(os.path.join(os.path.dirname(__file__), 'true_tag_info.json'), 
    lines=True)
dict_truetag_mapping = {}
for i in df_truetag_mapping.to_dict('records'):
    dict_truetag_mapping[i['true_tag_ids']] = i['category']
table_date = params['download_wish_truetag_tahoe_categories']['date']
output_jsonl = os.path.join(os.path.dirname(__file__), 'wishproducts_truetag_tahoe_stratsample.json')

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)

logging.info(
    F"Stratify Sample Wish Tahoe table [search.product_document_{table_date}]"
)

# %%
# Define the export table
export_table = {
    "name": "products_tmp",
    "columns": [
        {"name": "title", "type": "STRING"},
        {"name": "true_tag_ids", "type": "STRING"}
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
def insert_product_queries(table_date):
    """Insert top count user queries into table, along with gmv info"""
    q = f"""
    INSERT INTO {db}.{export_table['name']}
    SELECT title, true_tag_ids FROM (
    SELECT title, true_tag_ids, RANK() OVER (PARTITION BY true_tag_ids ORDER BY rand() DESC) AS row_number FROM (
        SELECT title, true_tag_ids
        FROM search.product_document_{table_date}
        GROUP BY true_tag_ids, title
        HAVING COUNT(*) > 10
        ORDER by rand()
        )
    )
    where row_number < 21
    """
    execute_async(q)

# %%
insert_product_queries(table_date)


# %%
# Show the data files for the external table
file_keys = get_s3_file_keys(s3_bucket=result_bucket, 
    s3_prefix="{}/{}".format(db, export_table["name"]))


# %%
# Import the parquet file as data frame
# Alternatively, you can use boto3 to directly download the file to disk
# For hundreds of data files, download using multiprocessing
logging.info(f'Write s3 parquet to {output_jsonl}')
with open(output_jsonl, 'w') as fout:
    for file_key, file_size in tqdm(file_keys):
        df_chunk = get_df_from_parquet(s3_bucket=result_bucket, s3_key=file_key)
        df_chunk['categories'] = df_chunk['true_tag_ids'].apply(
            lambda x: [dict_truetag_mapping[i] for i in x.split(',') if i in dict_truetag_mapping])
        for dat in df_chunk.to_dict('records'):
            fout.write(json.dumps(dat) + '\n')

# %%
# Optional: drop the external table
drop_external_table(
    db=db,
    table_name=export_table["name"],
    delete_files=True,
    s3_bucket=result_bucket,
    s3_prefix="{}/{}".format(db, export_table["name"]), 
)

# %%
# Show the data files for the external table
file_keys = get_s3_file_keys(s3_bucket=result_bucket, s3_prefix="{}/{}".format(db, export_table["name"]))
assert len(file_keys) == 0, "temp {} not deleted".format("{}/{}".format(db, export_table["name"]))

