# %%
import logging 
logging.basicConfig(level=logging.DEBUG)
import json
import gzip
import os
import pandas as pd
from tqdm import tqdm
from collections import Counter
import numpy as np
np.random.seed(42)
from sklearn.model_selection import train_test_split

# %%
current_folder = os.path.dirname(__file__)

tax_path = os.path.join(
    current_folder,
    '..',
    '..',
    'data',
    'taxonomy',
    'wish_newtax.json'
)

mturk_testset_path = os.path.join(
    current_folder,
    '..',
    '..',
    'data',
    'wish_products',
    'wish-mturk-labelled-09202022-clean-joinedlance.json'
)

train_path, val_path, test_path = [
    os.path.join(
        current_folder,
        '..',
        '..',
        'data',
        'wish_products_truetag_tahoe',
        f'Wish_Truetag_Tahoe_Meta_{i}_OpenaiInferred.json'
    ) for i in ['Train', 'Val', 'Test']
]

train_out_path, val_out_path, test_out_path = [
    os.path.join(
        current_folder,
        '..',
        '..',
        'data',
        'wish_products_truetag_tahoe',
        f'Wish_Truetag_Tahoe_Meta_{i}_OpenaiInferred_Processed.json'
    ) for i in ['Train', 'Val', 'Test']
]

df_tax = pd.read_json(tax_path, lines=True)
df_testset = pd.read_json(mturk_testset_path, lines=True)
test_titles = set(df_testset['title'].str.lower())
paths = set(df_tax['category_path'].str.lower())

df_train = pd.read_json(
    train_path, lines=True
)
df_val = pd.read_json(
    val_path, lines=True
)
df_test = pd.read_json(
    test_path, lines=True
)
df_train['split'] = 'train'
df_val['split'] = 'val'
df_test['split'] = 'test'

df = pd.concat([ 
    df_test,
    df_val,
    df_train
])

df = df[df.openai_category.apply(lambda x: ' > '.join(x) in paths)]

#%%
# prioritized keep
df.drop_duplicates(subset=['title'], keep="first", inplace=True, ignore_index=True)

df = df[df.title.apply(lambda x: x.lower() not in test_titles)]

logging.info("Processed split count: {}".format(Counter(df.split)))

#%%
df = df.rename(columns={
    "category": "truetag_category", 
    "text": "truetag_text", 
    "openai_category": "category"
})

df['text'] = df.title.str.strip() + ' -> ' + \
    df.category.apply(lambda x: ''.join(['[' + i.lower().strip() + ']' for i in x]))


# %%
df[df.split == 'train'][['title', 'category', 'text']].to_json(train_out_path, orient='records', lines=True)
df[df.split == 'val'][['title', 'category', 'text']].to_json(val_out_path, orient='records', lines=True)
df[df.split == 'test'][['title', 'category', 'text']].to_json(test_out_path, orient='records', lines=True)