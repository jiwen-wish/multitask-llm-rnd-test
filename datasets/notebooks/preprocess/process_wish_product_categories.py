#%% 
import logging
logging.basicConfig(level=logging.DEBUG)
import pickle 
import os
import gc
import json 

def dicts_to_json(dicts, filepath):
    with open(filepath, 'w') as f:
        for i in dicts:
            f.write(json.dumps(i))
            f.write('\n')

current_folder = os.path.dirname(__file__)

wish_pkl_path = os.path.join(
    current_folder,
    '..',
    '..',
    'data',
    'wish_products',
    'wishproducts.pkl'
)

train_path, val_path, test_path = [
    os.path.join(
        current_folder,
        '..',
        '..',
        'data',
        'wish_products',
        f'Wish_Meta_{i}.json'
    ) for i in ['Train', 'Val', 'Test']
]

train, valtest, cats = pickle.load(open(wish_pkl_path, 'rb'))
train = train.drop_duplicates(subset=['title'])
valtest = valtest.drop_duplicates(subset=['title'])
logging.info(f"{wish_pkl_path} loaded")
# %%
train['category'] = train['category'].apply(lambda x: [i.strip().lower() for i in x])
valtest['category'] = valtest['category'].apply(lambda x: [i.strip().lower() for i in x])

train['text'] = train.title.str.strip() + ' -> ' + \
    train.category.apply(lambda x: ''.join(['[' + i.lower().strip() + ']' for i in x]))
valtest['text'] = valtest.title.str.strip() + ' -> ' + \
    valtest.category.apply(lambda x: ''.join(['[' + i.lower().strip() + ']' for i in x]))

logging.info(f"Data transformed")
#%%
valtest_set = set(valtest.title)
train = train[train.title.apply(lambda x: x not in valtest_set)]
valtest = valtest.sample(frac=1.)
val = valtest.loc[:len(valtest)//2]
test = valtest.loc[len(valtest)//2:]
del valtest
gc.collect()

logging.info(f"Split created")

# %%
train = train[['title', 'category', 'text']].to_dict('records')
val = val[['title', 'category', 'text']].to_dict('records')
test = test[['title', 'category', 'text']].to_dict('records')

dicts_to_json(train, train_path)
dicts_to_json(val, val_path)
dicts_to_json(test, test_path)

logging.info(f"Exported to Json")