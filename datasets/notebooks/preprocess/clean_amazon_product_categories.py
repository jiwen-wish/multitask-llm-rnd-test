# %%
import pandas as pd
import os

# %%
current_folder = os.path.dirname(__file__)
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

train_clean_path, val_clean_path, test_clean_path = [
    os.path.join(
        current_folder,
        '..',
        '..',
        'data',
        'amazon',
        f'All_Amazon_Meta_{i}_Clean.json'
    ) for i in ['Train', 'Val', 'Test']
]

# %%
train = pd.read_json(train_path, lines=True)
val = pd.read_json(val_path, lines=True)
test = pd.read_json(test_path, lines=True)

def clean_cat(x):
    y = []
    for i in x:
        if '</span>' in i:
            break 
        else:
            y.append(i) 
    return y

cats = set(train.category.apply(tuple)).union(
    set(val.category.apply(tuple))).union(
    set(test.category.apply(tuple)))
cat_map = {i: tuple(clean_cat(i)) for i in cats}

print(f"Before clean: len(set(train.category)) = {len(set(train.category.apply(tuple)))}, "
f"len(set(val.category)) = {len(set(val.category.apply(tuple)))}, "
f"len(set(test.category)) = {len(set(test.category.apply(tuple)))}")

train['category'] = train['category'].apply(lambda x: list(cat_map[tuple(x)]))
val['category'] = val['category'].apply(lambda x: list(cat_map[tuple(x)]))
test['category'] = test['category'].apply(lambda x: list(cat_map[tuple(x)]))

print(f"After clean: len(set(train.category)) = {len(set(train.category.apply(tuple)))}, "
f"len(set(val.category)) = {len(set(val.category.apply(tuple)))}, "
f"len(set(test.category)) = {len(set(test.category.apply(tuple)))}")
#%%
print(f"Before clean: len(train) = {len(train)}, len(val) = {len(val)}, len(test) = {len(test)}")
train = train[(train.title.apply(lambda x: '->' not in x)) & (train.category.apply(len) > 0)]
val = val[val.title.apply(lambda x: '->' not in x) & (val.category.apply(len) > 0)]
test = test[test.title.apply(lambda x: '->' not in x) & (test.category.apply(len) > 0)]
print(f"After clean: len(train) = {len(train)}, len(val) = {len(val)}, len(test) = {len(test)}")

train['text'] = train['title'] + ' -> ' + train['category'].apply(lambda x: ''.join(['[' + i + ']' for i in x]))
val['text'] = val['title'] + ' -> ' + val['category'].apply(lambda x: ''.join(['[' + i + ']' for i in x]))
test['text'] = test['title'] + ' -> ' + test['category'].apply(lambda x: ''.join(['[' + i + ']' for i in x]))

# %%
train.to_json(train_clean_path, orient='records', lines=True)
val.to_json(val_clean_path, orient='records', lines=True)
test.to_json(test_clean_path, orient='records', lines=True)