# %%
import json
import gzip
import os
import pandas as pd
from tqdm import tqdm
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split

# %%
import dvc.api
params = dvc.api.params_show()

# %%
max_depth = params['process_amazon_categories']['max_depth']
max_unique_cats = params['process_amazon_categories']['max_unique_cats']

# %%
current_folder = os.path.dirname(__file__)
amazon_meta_path = os.path.join(
    current_folder,
    '..',
    '..',
    'data',
    'amazon',
    'All_Amazon_Meta.json.gz'
)

train_path, val_path, test_path = [
    os.path.join(
        current_folder,
        '..',
        '..',
        'data',
        'amazon',
        f'All_Amazon_Meta_{i}.json'
    ) for i in ['Train', 'Val', 'Test']
]

# %%
def num_there(s):
    return any(i.isdigit() for i in s)

def contains_bad_chars(s):
    s_set = set(s)
    bad_chars = '.@#$%^*()~+'
    for i in bad_chars:
        if i in s_set:
            return True
    return False


amazon_cats_counter = Counter()
data = {}
with gzip.open(amazon_meta_path) as f:
    for l in tqdm(f):
        dat = json.loads(l.strip())
        dat['category'] = [
            i.lower() for i in dat['category'] if \
                (not contains_bad_chars(i)) and \
                ('MP3' in i.upper() or '3D' in i.upper() or not num_there(i)) \
                and len(i) < 60 and len(i) > 1
        ]
        data[dat['title']] = dat['category']
        amazon_cats_counter.update(dat['category'])

print(F"{len(amazon_cats_counter)} unique cat names")
print(F"{len(data)} unique product names")
tmp = amazon_cats_counter.most_common()
amazon_cats_clean = set([i[0] for i in tmp if not contains_bad_chars(i[0])][:max_unique_cats])

for i in tqdm(data):
    data[i] = [j for j in data[i] if j in amazon_cats_clean]
data = {i: data[i][:max_depth] for i in data if len(data[i]) > 0}

df = pd.DataFrame(data.items(), columns=['title', 'category'])

df = df[(df.title.apply(len) > 0) & (df.category.apply(len) > 0)]

df['text'] = df.title.str.strip() + ' -> ' + \
    df.category.apply(lambda x: ''.join(['[' + i.lower().strip() + ']' for i in x]))

df = df.sample(frac=1, random_state=42)

train, _ = train_test_split(df, test_size=0.01, random_state=42)
val, test = train_test_split(_, test_size=0.5, random_state=42)

train.to_json(train_path, orient='records', lines=True)
val.to_json(val_path, orient='records', lines=True)
test.to_json(test_path, orient='records', lines=True)


