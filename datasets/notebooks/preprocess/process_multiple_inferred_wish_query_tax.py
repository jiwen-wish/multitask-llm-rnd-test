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
        f'Inferred_Multiple_Wish_Query_Meta_{i}.json'
    ) for i in ['Train', 'Val', 'Test']
]

inf_path = os.path.join(current_folder, '..', '..', 'data', 'query', 'wish_queries_inferred_newtax_multiple.json')
inf_path2 = os.path.join(current_folder, '..', '..', 'data', 'query', 'wish_queries_inferred_newtax.json')
tax_path = os.path.join(current_folder,  '..', '..', 'data', 'taxonomy', 'wish_newtax.json')
tmp = pd.read_json(inf_path2, lines=True)
tmp['search_table_name'] = 'query_new_category_inference_20220920'
df_inf = pd.concat([pd.read_json(inf_path, lines=True), tmp])
df_tax = pd.read_json(tax_path, lines=True)

df_inf = df_inf.sort_values('search_table_name', ascending=False).drop_duplicates(
    subset='query', keep='first')

#%%
id2path = {}
for i in df_tax.to_dict('records'):
    id2path[str(i['id'])] = i['category_path']

df_inf['category_paths'] = df_inf['categories'].apply(lambda x: [id2path[i] for i in x.split(',')])
df_inf['category_weights'] = df_inf['weights'].apply(lambda x: [float(i) for i in x.split(',')])

#%%
df_out = df_inf[['query', 'category_paths']].copy()
df_out = df_out.sample(frac=1, random_state=42)

train, _ = train_test_split(df_out, test_size=0.1, random_state=42)
val, test = train_test_split(_, test_size=0.5, random_state=42)

#%%
recs  = []
for i in train.to_dict('records'):
    for j in i['category_paths']:
        recs.append({'query': i['query'], 'category': j.lower().strip()})
train = pd.DataFrame(recs)

recs  = []
for i in val.to_dict('records'):
    for j in i['category_paths']:
        recs.append({'query': i['query'], 'category': j.lower().strip()})
val = pd.DataFrame(recs)

recs  = []
for i in test.to_dict('records'):
    for j in i['category_paths']:
        recs.append({'query': i['query'], 'category': j.lower().strip()})
test = pd.DataFrame(recs)

train.to_json(train_path, orient='records', lines=True)
val.to_json(val_path, orient='records', lines=True)
test.to_json(test_path, orient='records', lines=True)
