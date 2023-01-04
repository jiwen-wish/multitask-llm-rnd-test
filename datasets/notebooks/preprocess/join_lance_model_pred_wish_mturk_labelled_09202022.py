#%% 
from tahoe import create_table, create_external_table, drop_external_table, execute_async
from s3 import result_bucket, temp_bucket, get_s3_file_keys, upload_df_to_parquet, get_df_from_parquet

import pandas as pd
import logging
import os

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)

current_folder = os.path.dirname(__file__)

wish_mturk_test_path = os.path.join(
    current_folder,
    '..',
    '..',
    'data',
    'wish_products',
    'wish-mturk-labelled-09202022-clean.json'
)

wish_mturk_test_joined_path = os.path.join(
    current_folder,
    '..',
    '..',
    'data',
    'wish_products',
    'wish-mturk-labelled-09202022-clean-joinedlance.json'
)

db = "sweeper_dev"
df_data = pd.read_json(wish_mturk_test_path, lines=True)
df_data = df_data.drop_duplicates(subset=['pid'])

#%%
# Create the temp import table
# We do not partition the external table to avoid the additional effort for adding data to specific partitions and register the partitions
temp_test_table = {
    "name": "mturk_wish_products_labelled_tmp",
    "columns": [
        {"name": "pid", "type": "STRING"},
        {"name": "title", "type": "STRING"},
        {"name": "category", "type": "ARRAY<STRING>"},
        {"name": "text", "type": "STRING"},
    ]
}

drop_external_table(
    db=db,
    table_name=temp_test_table["name"],
    delete_files=True,
    s3_bucket=temp_bucket,
    s3_prefix=f'{db}/{temp_test_table["name"]}', 
)

create_external_table(
    table_name=temp_test_table["name"],
    table_definition=temp_test_table,
    db=db,
    bucket=temp_bucket
)

upload_df_to_parquet(df_data, s3_bucket=temp_bucket, s3_key=f"{db}/{temp_test_table['name']}/data.parquet")

#%% 
# Join lance model pred

export_table = {
    "name": "mturk_wish_products_labelled_joined_tmp",
    "columns": [
        {"name": "pid", "type": "STRING"},
        {"name": "title", "type": "STRING"},
        {"name": "category", "type": "ARRAY<STRING>"},
        {"name": "text", "type": "STRING"},
        {"name": "lance_predicted_category_id", "type": "INTEGER"},
        {"name": "lance_predicted_category","type": "STRING"},
        {"name": "lance_predicted_is_leaf", "type": "BOOLEAN"}
    ]
}

drop_external_table(
    db=db,
    table_name=export_table["name"],
    delete_files=True,
    s3_bucket=result_bucket,
    s3_prefix=f"{db}/{export_table['name']}"
)

create_external_table(
    table_name=export_table["name"],
    table_definition=export_table,
    db=db,
    bucket=result_bucket
)

#%%
# Export data to the external table
q = f"""
INSERT INTO {db}.{export_table['name']}
SELECT 
    A.pid, A.title, A.category, A.text, B.category_id as "lance_predicted_category_id", 
    C.category_path as "lance_predicted_category", C.is_leaf as "lance_predicted_is_leaf"
FROM {db}.{temp_test_table['name']}  A
LEFT JOIN structured_data.latest_listing_category_predictions B
ON A.pid = B.listing_id 
LEFT JOIN structured_data.wish_product_categories C 
ON B.category_id = C.id
"""
execute_async(q)

#%%
file_keys = get_s3_file_keys(s3_bucket=result_bucket, s3_prefix=f"{db}/{export_table['name']}")

dfs = []
for file_key, file_size in file_keys:
    df_chunk = get_df_from_parquet(s3_bucket=result_bucket, s3_key=file_key)
    dfs.append(df_chunk)
    
df_data = pd.concat(dfs)
df_data['lance_predicted_category'] = df_data['lance_predicted_category'].apply(
    lambda x: [i.strip().lower() for i in x.split(' > ')]
)
df_data.to_json(wish_mturk_test_joined_path, orient='records', lines=True)

#%%
drop_external_table(
    db=db,
    table_name=temp_test_table["name"],
    delete_files=True,
    s3_bucket=temp_bucket,
    s3_prefix=f'{db}/{temp_test_table["name"]}', 
)

drop_external_table(
    db=db,
    table_name=export_table["name"],
    delete_files=True,
    s3_bucket=result_bucket,
    s3_prefix=f"{db}/{export_table['name']}"
)
