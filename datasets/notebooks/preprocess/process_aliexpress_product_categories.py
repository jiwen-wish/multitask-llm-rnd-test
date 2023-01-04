#%% 
import pickle 
import os

current_folder = os.path.dirname(__file__)

aliexpress_pkl_path = os.path.join(
    current_folder,
    '..',
    '..',
    'data',
    'aliexpress',
    'aliexpress_train_val_test_cat.pkl'
)

train_path, val_path, test_path = [
    os.path.join(
        current_folder,
        '..',
        '..',
        'data',
        'aliexpress',
        f'Aliexpress_Meta_{i}.json'
    ) for i in ['Train', 'Val', 'Test']
]

train, val, test, cats = pickle.load(open(aliexpress_pkl_path, 'rb'))
# %%
cats['category'] = cats['category_path'].apply(lambda x: [i.strip().lower() for i in x.strip().split(' > ')])
cats = cats.rename(columns={'id': 'category_id'})

train = train.merge(cats[['category_id', 'category']], on='category_id', how='inner')
train = train[['title', 'category']]
train['text'] = train.title.str.strip() + ' -> ' + \
    train.category.apply(lambda x: ''.join(['[' + i.lower().strip() + ']' for i in x]))

val = val.merge(cats[['category_id', 'category']], on='category_id', how='inner')
val = val[['title', 'category']]
val['text'] = val.title.str.strip() + ' -> ' + \
    val.category.apply(lambda x: ''.join(['[' + i.lower().strip() + ']' for i in x]))

test = test.merge(cats[['category_id', 'category']], on='category_id', how='inner')
test = test[['title', 'category']]
test['text'] = test.title.str.strip() + ' -> ' + \
    test.category.apply(lambda x: ''.join(['[' + i.lower().strip() + ']' for i in x]))

#%%
train = train.drop_duplicates(subset=['title'])
val = val.drop_duplicates(subset=['title'])
test = test.drop_duplicates(subset=['title'])
test_set = set(test.title)
val = val[val.title.apply(lambda x: x not in test_set)]
val_set = set(val.title)
train = train[train.title.apply(lambda x: x not in test_set and x not in val_set)]

# %%
train.to_json(train_path, orient='records', lines=True)
val.to_json(val_path, orient='records', lines=True)
test.to_json(test_path, orient='records', lines=True)
