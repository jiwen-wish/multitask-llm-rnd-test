#%%

from tahoe import create_external_table, drop_external_table, execute_async
from s3 import result_bucket, get_s3_file_keys, get_df_from_parquet
import pandas as pd
import numpy as np
import json
import logging
import os
import dvc.api
from tqdm import tqdm

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)

params = dvc.api.params_show()

country_search_ratios = params['download_latest_wish_queries_join_v2_english']['country_search_ratios']

table_res = execute_async(
    query="""SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'search'
AND table_name like 'query_join_v2_%'
ORDER BY date_parse(
  regexp_replace(regexp_extract(table_name, 'v2_(\d+)'), 'v2_'), 
  '%Y%m%d%H%i%S'
) DESC LIMIT 1"""
)
# %%
assert len(table_res) == 1 and table_res[0][0].startswith('query_join_v2_')
table_name = table_res[0][0]

output_jsonl = os.path.join(os.path.dirname(__file__), "joinv2_queries_en_{}.json".format(
    table_name.replace('query_join_v2_', '')
))
# %%
export_table = {
    "name": "query_joinv2_export_tmp",
    "columns": [
        {"name": "query", "type": "STRING"},
        {"name": "avg_price", "type": "FLOAT"},
        {"name": "p25_price", "type": "FLOAT"},
        {"name": "p50_price", "type": "FLOAT"},
        {"name": "p75_price", "type": "FLOAT"},
        {"name": "true_tags", "type": "STRING"},
        {"name": "tag_weights", "type": "STRING"},
        {"name": "search_gmv", "type": "FLOAT"},
        {"name": "tot_buyers", "type": "INTEGER"},
        {"name": "tot_searches", "type": "INTEGER"},
        {"name": "tot_searchers", "type": "INTEGER"},
        {"name": "frequency_group", "type": "STRING"}, 
        {"name": "total_product_gmv", "type": "FLOAT"}, 
        {"name": "brand_product_gmv", "type": "FLOAT"}, 
        {"name": "we_product_gmv", "type": "FLOAT"}, 
        {"name": "gmv_by_country", "type": "ARRAY<STRING>"},
        {"name": "buyers_by_country", "type": "ARRAY<STRING>"},
        {"name": "searches_by_country", "type": "ARRAY<STRING>"},
        {"name": "searchers_by_country", "type": "ARRAY<STRING>"},
        {"name": "new_categories", "type": "STRING"},
        {"name": "new_category_weights", "type": "STRING"},
    ]
}
db = "sweeper_dev"

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
def insert_join_queries(table_name, country_search_ratios):
    """Insert data for english queries, at least 10% searches come from 4 main english speaking countries"""
    q = f"""
    INSERT INTO {db}.{export_table['name']}
    SELECT query, avg_price, p25_price, p50_price, p75_price, 
        true_tags, tag_weights, search_gmv, tot_buyers, tot_searches, 
        tot_searchers, frequency_group, total_product_gmv, brand_product_gmv, 
        we_product_gmv, gmv_by_country, buyers_by_country, searches_by_country, 
        searchers_by_country, new_categories, new_category_weights FROM search.{table_name}
    WHERE cardinality(
    filter(
        searches_by_country, 
        x -> (
            COUNTRY_SEARCH_RATIOS
            )
        )
    ) > 0
    """.replace('COUNTRY_SEARCH_RATIOS', 
        ' OR '.join([
            f"(x like '{k}%' and CAST(regexp_extract(x, '\d+') AS REAL) / tot_searches > {v})" \
                for k, v in country_search_ratios.items()
        ])
    )
    execute_async(q)

# %%
insert_join_queries(table_name, country_search_ratios)

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

with open(output_jsonl, 'w') as f:
    for file_key, file_size in file_keys:
        df_chunk = get_df_from_parquet(s3_bucket=result_bucket, s3_key=file_key)
        for i in tqdm(df_chunk.to_dict('records')):
            for k in i:
                if isinstance(i[k], np.ndarray):
                    i[k] = i[k].tolist()
            for k in i:
                if not isinstance(i[k], list):
                    if pd.isna(i[k]):
                        i[k] = None
            f.write(json.dumps(i) + '\n')
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



