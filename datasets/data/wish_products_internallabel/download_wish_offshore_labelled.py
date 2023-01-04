# %%
from tahoe import create_external_table, drop_external_table, execute_async
from s3 import result_bucket, get_s3_file_keys, get_df_from_parquet
import pandas as pd
import logging
import os
import dvc.api

params = dvc.api.params_show()
output_jsonl = os.path.join(os.path.dirname(__file__), 'wish_products_offshore_labelled.json')

table_date = params['wish_products_offshore_join_product_tahoe']['date']

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)

logging.info(
    (F"Fetch Wish products labelled by offshore team"
        F" and save to {output_jsonl}")
)

# %%
# Define the export table
export_table = {
    "name": "products_offshore_export_tmp",
    "columns": [
        {"name": "pid", "type": "STRING"},
        {"name": "labeler_leaf_id", "type": "STRING"},
        {"name": "labeler_id_path", "type": "STRING"},
        {"name": "model_leaf_id	", "type": "STRING"},
        {"name": "model_id_path", "type": "STRING"},
        {"name": "title", "type": "STRING"},
        {"name": "product_description", "type": "STRING"},
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
    s3_prefix="{}/{}".format(db, export_table["name"]), 
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
def insert_data(table_date):
    """Insert top count user queries into table, along with gmv info"""
    q = f"""
    INSERT INTO {db}.{export_table['name']}
    SELECT ttt.pid, ttt.labeler_leaf_id, ttt.labeler_id_path, CAST(ttt.model_leaf_id as varchar), ttt.model_id_path, 
    search.product_document_{table_date}.title, search.product_document_{table_date}.product_description, search.product_document_{table_date}.true_tag_ids
    FROM 
    (
    SELECT DISTINCT 
    tt.pid, 
    tt.labeler_id AS labeler_leaf_id, 
    l.category_id_path AS labeler_id_path,
    p.category_id AS model_leaf_id, 
    m.category_id_path AS model_id_path
    FROM (
        SELECT 
        pid_list[1] AS pid, 
        job_id, 
        category_id AS labeler_id
        FROM (
        SELECT 
        regexp_extract_all(image_urls, '[0-9][a-zA-Z0-9]+{{6,}}') AS pid_list, 
        s.job_id, 
        t.category_id 
        FROM sweeper.category_classification_events s
        JOIN (
            SELECT job_id, category_id
            FROM sweeper.category_classification_events
            WHERE category_id IS NOT NULL) t on t.job_id = s.job_id
        WHERE s.title IS NOT NULL
        -- AND t.category_id IS NOT NULL
        ) d
        WHERE cardinality(pid_list) >= 1
    ) tt
    JOIN structured_data.latest_listing_category_predictions p ON p.listing_id = tt.pid
    JOIN structured_data.wish_product_categories m ON m.id = p.category_id
    JOIN structured_data.wish_product_categories l ON l.id = CAST(tt.labeler_id as int)
    )ttt LEFT JOIN search.product_document_{table_date} ON ttt.pid = search.product_document_{table_date}.id
    """
    execute_async(q)

# %%
insert_data(table_date)

# %%
# Check again the number of rows in the external table
q = f"SELECT COUNT(*) FROM {db}.{export_table['name']}"
execute_async(q)

# %%
# Show the data files for the external table
file_keys = get_s3_file_keys(s3_bucket=result_bucket, 
    s3_prefix="{}/{}".format(db, export_table["name"]))

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
    s3_prefix="{}/{}".format(db, export_table["name"]), 
)

# %%
# Show the data files for the external table
file_keys = get_s3_file_keys(s3_bucket=result_bucket, s3_prefix="{}/{}".format(db, export_table["name"]))
assert len(file_keys) == 0, "temp {} not deleted".format("{}/{}".format(db, export_table["name"]))

# %%



