# %%
import os
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

# %%
current_folder = os.path.dirname(__file__)

train_path, val_path, test_path = [
    os.path.join(
        current_folder,
        '..',
        '..',
        'data',
        'amazon_query',
        f'Amazon_ESCI_Query_{i}.json'
    ) for i in ['Train', 'Val', 'Test']
]

tax_train_path, tax_val_path, tax_test_path = [
    os.path.join(
        current_folder,
        '..',
        '..',
        'data',
        'amazon',
        f'All_Amazon_Meta_{i}_Clean.json'
    ) for i in ['Train', 'Val', 'Test']
]

esci_label2gain = {
    'E' : 1.0,
    'S' : 0.1,
    'C' : 0.01,
    'I' : 0.0,
}

df_examples = pd.read_parquet(os.path.join(current_folder, '..', '..', 'data', 'amazon_query', 'shopping_queries_dataset_examples.parquet'))
df_products = pd.read_parquet(os.path.join(current_folder, '..', '..', 'data', 'amazon_query', 'shopping_queries_dataset_products.parquet'))

# %%
df_examples_products = pd.merge(
    df_examples,
    df_products,
    how='left',
    left_on=['product_locale','product_id'],
    right_on=['product_locale', 'product_id']
)
# %%
df_task_1_large = df_examples_products
print(f"Amazon ESCI Task 1 Full large set(Ranking) has {len(set(df_task_1_large.product_title))} titles, {len(set(df_task_1_large['query']))} queries")

# %%
df_amazon_train = pd.read_json(tax_train_path, lines=True)
df_amazon_val = pd.read_json(tax_val_path, lines=True)
df_amazon_test = pd.read_json(tax_test_path, lines=True)

# %%
df_amazon_train['cat_split'] = 'train'
df_amazon_val['cat_split'] = 'val'
df_amazon_test['cat_split'] = 'test'

# %%
df_amazon_concat = pd.concat([ 
    df_amazon_train,
    df_amazon_val, 
    df_amazon_test
])

# %%
df_amazon_cat_merge = df_amazon_concat[['title', 'category', 'cat_split']].merge(df_task_1_large.rename(columns={'product_title': 'title'}), on='title', how='right')

# %%
df_amazon_cat_merge = df_amazon_cat_merge.where(pd.notnull(df_amazon_cat_merge), None)

# %%
print("Merging with Amazon title2tax data split_counter", Counter(df_amazon_cat_merge[['cat_split', 'split']].to_records(index=False).tolist()))

# %%
df_amazon_cat_merge_clean = df_amazon_cat_merge[~(((df_amazon_cat_merge.cat_split == 'test') & (df_amazon_cat_merge.split == 'train')) | \
    ((df_amazon_cat_merge.split == 'test') & (df_amazon_cat_merge.cat_split == 'train')) | \
    ((df_amazon_cat_merge.split == 'test') & (df_amazon_cat_merge.cat_split == 'val'))
)]

# %%
print(f"{1 - len(df_amazon_cat_merge_clean) / len(df_amazon_cat_merge)} of data has data-leak problem, ignore")

df_amazon_cat_merge_clean = df_amazon_cat_merge

# %%
df_amazon_cat_merge_clean['labels'] = df_amazon_cat_merge_clean['esci_label'].apply(lambda x: esci_label2gain[x])

# %%
df_amazon_cat_merge_clean_simple = df_amazon_cat_merge_clean[['title', 'query', 'category', 'example_id', 'query_id', 'product_id', 'product_locale', 
    'esci_label', 'labels', 'small_version', 'split']]

# %%
df_amazon_cat_merge_clean_simple['text_input'] = df_amazon_cat_merge_clean_simple['title']
df_amazon_cat_merge_clean_simple['text_output'] = df_amazon_cat_merge_clean_simple['query']

# %%
train = df_amazon_cat_merge_clean_simple[df_amazon_cat_merge_clean_simple.split == 'train']
train, val =  train_test_split(train, test_size=0.2, random_state=42)
test = df_amazon_cat_merge_clean_simple[df_amazon_cat_merge_clean_simple.split == 'test']

# %%
print(f"len(train): {len(train)}, len(val): {len(val)}, len(test): {len(test)}")

# %%
print(f"train.labels.mean(): {train.labels.mean()}, val.labels.mean(): {val.labels.mean()}, test.labels.mean(): {test.labels.mean()}")

train.to_json(train_path, orient='records', lines=True)
val.to_json(val_path, orient='records', lines=True)
test.to_json(test_path, orient='records', lines=True)

