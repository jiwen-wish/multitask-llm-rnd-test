import logging
import pandas as pd
from s3 import get_s3_file_keys, get_s3_client
from collections import defaultdict, Counter
from tahoe import execute_async
import os
from pathlib import Path
import glob
import pickle
import shutil


data_dir = os.path.dirname(__file__)
taxonomy_path = f"{data_dir}/wish_categories.csv"
validation_df_path = f"{data_dir}/validation_texts.parquet"
training_chunks_dir = f"{data_dir}/training_chunks"
Path(training_chunks_dir).mkdir(exist_ok=True)
pickle_path = f"{data_dir}/wishproducts.pkl"

def download_texts():
    sdt_bucket = "structured-data-dev"
    train_chunks_prefix = "category_model_texts_2022q1/training_chunks/"
    valid_df_key = "category_model_texts_2022q1/validation_texts.parquet"
    data_file_keys = get_s3_file_keys(
        s3_bucket=sdt_bucket,
        s3_prefix=train_chunks_prefix
    )
    logging.info(f"Number of data files: {len(data_file_keys)}")

    s3 = get_s3_client()
    s3.download_file(sdt_bucket, valid_df_key, validation_df_path)
    logging.info(f"Downloaded: {validation_df_path}")

    for file_key, file_size in data_file_keys:
        file_name = file_key[file_key.rfind("/") + 1:]
        out_path = f"{training_chunks_dir}/{file_name}"
        s3.download_file(sdt_bucket, file_key, out_path)
        logging.info(f"Downloaded: {out_path}")


def download_categories():
    version = "v1.0.0"
    columns = [
        "id",
        "name",
        "level",
        "is_leaf",
        "parent_id"
    ]
    q = f"""
    SELECT {", ".join(columns)}
    FROM structured_data_stage.wish_category_snapshots
    WHERE category_tree_version = '{version}'
        AND name != 'Root'
    ORDER BY id
    """
    records = execute_async(q)
    df_categories = pd.DataFrame(records, columns=columns)
    logging.info(f"Categories download: {df_categories.shape}")
    # Generate ancestor_id
    child2parent = {row.id: row.parent_id for row in df_categories.itertuples()}
    def get_ancestor(cat_id):
        ancestor_id = cat_id
        while child2parent[ancestor_id] != 1:
            ancestor_id = child2parent[ancestor_id]
        return ancestor_id
    df_categories["ancestor_id"] = df_categories["id"].apply(get_ancestor)
    df_categories.to_csv(taxonomy_path, index=False)
    logging.info(f"Categories written to {taxonomy_path}")


def process_data():
    logging.info("Process data")
    # read category
    cats = pd.read_csv(taxonomy_path)
    cats['id'] = cats['id'].astype('int')
    cats['level'] = cats['level'].astype('int')
    cats['parent_id'] = cats['parent_id'].astype('int')
    cats['ancestor_id'] = cats['ancestor_id'].astype('int')
    cats = cats.drop_duplicates(subset=['id'])
    assert len(cats[cats.name.apply(lambda x: '>' in x or '[' in x or ']' in x)]) == 0

    # create paths
    id2cat = {}
    id2parent = {}
    for i in cats.to_dict('records'):
        id2cat[i['id']] = i['name']
        id2parent[i['id']] = i['parent_id']

    recs = []
    for i in cats.to_dict('records'):
        i['category'] = [i['name']]
        parent = i['parent_id']
        
        while parent != 1:
            i['category'].insert(0, id2cat[parent])
            parent = id2parent[parent]
        recs.append(i)
    cats = pd.DataFrame(recs)
    logging.info("Category paths created")
    # inner join
    df_val = pd.read_parquet(validation_df_path)
    df_val = df_val.drop_duplicates('id')
    df_train = pd.concat( 
        pd.read_parquet(i) for i in glob.glob(f"{training_chunks_dir}/texts.parquet*")
    )
    df_train = df_train.drop_duplicates('id')
    df_train = df_train.merge(cats.rename(columns={'id': 'category_id'}), on='category_id', how='inner')
    df_val = df_val.merge(cats.rename(columns={'id': 'category_id'}), on='category_id', how='inner')
    pickle.dump(
        (df_train, df_val, cats),
        open(pickle_path, 'wb')
    )
    logging.info("train_val_cats Pickle saved")

def clean_dir():
    os.remove(validation_df_path)
    os.remove(taxonomy_path)
    shutil.rmtree(training_chunks_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO
    )
    download_categories()
    download_texts()
    process_data()
    clean_dir()