#%%
import os 
import pandas as pd

current_folder = os.path.dirname(__file__)

amazon_train_path, amazon_val_path, amazon_test_path = [
    os.path.join(
        current_folder,
        '..',
        '..',
        'data',
        'amazon',
        f'All_Amazon_Meta_{i}_Clean.json'
    ) for i in ['Train', 'Val', 'Test']
]

mave_pos_path = os.path.join(
    current_folder,
    '..',
    '..',
    'data',
    'mave_attr_extract',
    'mave_positives_titleonly_genreformat.json'
)

mave_neg_path = os.path.join(
    current_folder,
    '..',
    '..',
    'data',
    'mave_attr_extract',
    'mave_negatives_titleonly_genreformat.json'
)

mave_train_path, mave_val_path, mave_test_path = [
    os.path.join(
        current_folder,
        '..',
        '..',
        'data',
        'mave_attr_extract',
        f'Mave_Title_GENRE_{i}.json'
    ) for i in ['Train', 'Val', 'Test']
]

#%%
df_mave_pos = pd.read_json(mave_pos_path, lines=True).drop_duplicates('title')
df_mave_neg = pd.read_json(mave_neg_path, lines=True).drop_duplicates('title')

#%% 
df_amazon_train = pd.read_json(amazon_train_path, lines=True)
df_amazon_val = pd.read_json(amazon_val_path, lines=True)
df_amazon_test = pd.read_json(amazon_test_path, lines=True)

#%%
df_mave_pos['label_pos_neg'] = 'pos'
df_mave_neg['label_pos_neg'] = 'neg'
df_mave = pd.concat([df_mave_pos, df_mave_neg]).drop_duplicates('title')

#%%
df_mave_train = df_amazon_train[['title', 'category']].merge(
    df_mave, on='title', how='inner')
df_mave_val = df_amazon_val[['title', 'category']].merge(
    df_mave, on='title', how='inner')
df_mave_test = df_amazon_test[['title', 'category']].merge(
    df_mave, on='title', how='inner')

#%%
matched_mave_titles = set(df_mave_train.title).union(
    set(df_mave_val.title)).union(
    set(df_mave_test.title))
df_mave_rest = df_mave[df_mave.title.apply(
    lambda x: x not in matched_mave_titles)]

# %%
print("Matched {} with train, {} with val, {} with test, and merge the rest {} to train".format(
    len(df_mave_train), len(df_mave_val), len(df_mave_test), len(df_mave_rest)
))

#%% 
df_mave_rest['category'] = None 
df_mave_train_merge = pd.concat([df_mave_train, df_mave_rest])

#%% 
df_mave_train_merge.to_json(mave_train_path, orient='records', lines=True)
df_mave_val.to_json(mave_val_path, orient='records', lines=True)
df_mave_test.to_json(mave_test_path, orient='records', lines=True)