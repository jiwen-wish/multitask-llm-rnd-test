#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
import os
#%%
current_folder = os.path.dirname(__file__)

train_path, val_path, test_path = [
    os.path.join(
        current_folder,
        '..',
        '..',
        'data',
        'query',
        f'Inferred_Wish_Query_Meta_{i}.json'
    ) for i in ['Train', 'Val', 'Test']
]

inf_path = os.path.join(current_folder, '..', '..', 'data', 'query', 'wish_queries_inferred_newtax.json')
tax_path = os.path.join(current_folder,  '..', '..', 'data', 'taxonomy', 'wish_newtax.json')

df_inf = pd.read_json(
    inf_path, 
    lines=True)
df_tax = pd.read_json(
    tax_path, 
    lines=True)
#%%
id2path = {}
for i in df_tax.to_dict('records'):
    id2path[str(i['id'])] = i['category_path']

df_inf['category_paths'] = df_inf['categories'].apply(lambda x: [id2path[i] for i in x.split(',')])

df_out = df_inf[['query', 'category_paths']].copy()

df_out['category'] = df_out['category_paths'].apply(lambda x: [i.strip().lower() for i in x[0].split(' > ')])

df_out['text'] = df_out['query'].str.strip() + ' -> ' + \
    df_out.category.apply(lambda x: ''.join(['[' + i + ']' for i in x]))

df_out = df_out[['query', 'category', 'text']]

df_out = df_out.drop_duplicates(subset='query')

df_out = df_out.sample(frac=1, random_state=42)

train, _ = train_test_split(df_out, test_size=0.1, random_state=42)
val, test = train_test_split(_, test_size=0.5, random_state=42)

train.to_json(train_path, orient='records', lines=True)
val.to_json(val_path, orient='records', lines=True)
test.to_json(test_path, orient='records', lines=True)
