# %%
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
meta_path = os.path.join(
    current_folder,
    '..',
    '..',
    'data',
    'wish_products_truetag_tahoe',
    'wishproducts_truetag_tahoe.json.gz'
)

train_path, val_path, test_path = [
    os.path.join(
        current_folder,
        '..',
        '..',
        'data',
        'wish_products_truetag_tahoe',
        f'Wish_Truetag_Tahoe_Meta_{i}.json'
    ) for i in ['Train', 'Val', 'Test']
]

# %%
data = {}
num_products = 0
num_lines = 0
with gzip.open(meta_path) as f, \
    open(train_path, 'w') as train_f, \
    open(val_path, 'w') as val_f, \
    open(test_path, 'w') as test_f:
    for l in tqdm(f):
        num_lines += 1
        dat = json.loads(l.strip())
        data[dat['title'].replace('->', '')] = dat['categories']
        if len(data) > 10000000:
            num_products += len(data)
            for k, vs in data.items():
                for v in vs:
                    if len(v) > 0:
                        rec = {'title': k, 'category': v,
                            'text': k.strip() + ' -> ' + ''.join(['[' + i.lower().strip() + ']' for i in v])}
                        rec_json = json.dumps(rec)
                        if np.random.random() < .9999:
                            train_f.write(rec_json)
                            train_f.write('\n')
                        elif np.random.random() < .5:
                            val_f.write(rec_json)
                            val_f.write('\n')
                        else:
                            test_f.write(rec_json)
                            test_f.write('\n')
            data = {}
    if len(data):
        num_products += len(data)
        for k, vs in data.items():
            for v in vs:
                if len(v) > 0:
                    rec = {'title': k, 'category': v,
                        'text': k.strip() + ' -> ' + ''.join(['[' + i.lower().strip() + ']' for i in v])}
                    rec_json = json.dumps(rec)
                    if np.random.random() < .9999:
                        train_f.write(rec_json)
                        train_f.write('\n')
                    elif np.random.random() < .5:
                        val_f.write(rec_json)
                        val_f.write('\n')
                    else:
                        test_f.write(rec_json)
                        test_f.write('\n')
        data = {}

print(F"{num_products} product titles processed from {num_lines} lines")