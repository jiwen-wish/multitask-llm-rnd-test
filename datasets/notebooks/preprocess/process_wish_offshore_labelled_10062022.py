#%%
import pandas as pd
import os

current_folder = os.path.dirname(__file__)

wish_offshore_file_path = os.path.join(
    current_folder,
    '..',
    '..',
    'data',
    'wish_products_internallabel',
    'wish_products_offshore_labelled.json'
)

wish_offshore_test_path = os.path.join(
    current_folder,
    '..',
    '..',
    'data',
    'wish_products_internallabel',
    'wish_products_offshore_labelled_processed.json'
)

tax_path = os.path.join(
    current_folder,
    '..',
    '..',
    'data',
    'taxonomy',
    'wish_newtax.json'
)

df = pd.read_json(wish_offshore_file_path, lines=True)
df_tax = pd.read_json(tax_path, lines=True)

#%%
id2path = {}
id2node = {}
for i in df_tax.to_dict('records'):
    id2path[i['id']] = i['category_path'].lower()
    id2node[i['id']] = i['category_path'].split(' > ')[-1].lower().strip()

assert (df['labeler_leaf_id'].apply(lambda x: id2path[x]) == \
    df['labeler_id_path'].apply(lambda x: ' > '.join(
        [id2node[int(i)] for i in x.split(',')])
    )).all()
# %%
df['category'] = df['labeler_id_path'].apply(
    lambda x: [id2node[int(i)] for i in x.split(',')])
df['text'] = df['title'] + ' -> ' + df['category'].apply(
    lambda x: ''.join(['[' + i + ']' for i in x]))
# %%
df['lance_predicted_category'] = df['model_id_path'].apply(
    lambda x: [id2node[int(i)] for i in x.split(',')])

# %%
df = df[~df.text.isna()]
df = df.drop_duplicates('title')

df.to_json(wish_offshore_test_path, lines=True, orient='records')