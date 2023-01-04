# %%
from tahoe import create_external_table, drop_external_table, execute_async
from s3 import result_bucket, get_s3_file_keys, get_df_from_parquet
import pandas as pd
import logging
import dvc.api
import os
import gzip
from tqdm import tqdm
import json

params = dvc.api.params_show()

df_truetag_mapping = pd.read_json(os.path.join(os.path.dirname(__file__), '..', 
    'wish_products_truetag_tahoe', 'true_tag_info.json'), lines=True)
dict_truetag_mapping = {}
for i in df_truetag_mapping.to_dict('records'):
    dict_truetag_mapping[i['true_tag_ids']] = i['category']


output_jsonlgz = os.path.join(os.path.dirname(__file__), 'wishproducts_clipmore_tahoe.json.gz')
table_date = params['download_wish_truetag_tahoe_categories']['date']

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)


# # %%
# Define the export table
export_table = {
    "name": "product_clip_tmp",
    "columns": [
        {"name": "product_id", "type": "STRING"},
        {"name": "img_embedding", "type": "STRING"},
        {"name": "title", "type": "STRING"},
        {"name": "product_description", "type": "STRING"},
        {"name": "product_merchant_tags", "type": "STRING"},
        {"name": "true_tag_ids", "type": "STRING"},
        {"name": "true_tags_are_predicted", "type": "STRING"},
        {"name": "category_name", "type": "STRING"},
        {"name": "category_id_path", "type": "STRING"}
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
def insert_clip():
    """Insert clip"""
    q = f"""
    INSERT INTO {db}.{export_table['name']}
    SELECT c.product_id, c.img_embedding, p.title, p.product_description, p.product_merchant_tags, p.true_tag_ids, 
    p.true_tags_are_predicted, p.category_name, p.category_id_path from 
    supply.microtagging_master_image_embeds c INNER JOIN search.product_document_{table_date} p
    ON c.product_id = p.id
    """
    execute_async(q)

# %%
# insert_top_queries(week_ago_dt)
insert_clip()

# %%
# Check again the number of rows in the external table
q = f"SELECT COUNT(*) FROM {db}.{export_table['name']}"
execute_async(q)

# %%
# Show the data files for the external table
file_keys = get_s3_file_keys(s3_bucket=result_bucket, 
    s3_prefix="sweeper_dev/{}".format(export_table["name"]))

# %%
# Import the parquet file as data frame
# Alternatively, you can use boto3 to directly download the file to disk
# For hundreds of data files, download using multiprocessing

logging.info(f'Write s3 parquet to {output_jsonlgz}')
with gzip.open(output_jsonlgz, 'w') as fout:
    for file_key, file_size in tqdm(file_keys):
        df_chunk = get_df_from_parquet(s3_bucket=result_bucket, s3_key=file_key)
        df_chunk['categories'] = df_chunk['true_tag_ids'].apply(
            lambda x: [dict_truetag_mapping[i] for i in x.split(',') if i in dict_truetag_mapping])
        for dat in df_chunk.to_dict('records'):
            fout.write((json.dumps(dat) + '\n').encode('utf-8'))

# %%
try:
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
except Exception as e:
    print(e)
    print(f'{db}.{export_table["name"]} Table not dropped, need to manually drop')




