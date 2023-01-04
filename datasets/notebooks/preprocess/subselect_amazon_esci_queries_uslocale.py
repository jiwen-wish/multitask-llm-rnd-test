# %%
import os
import pandas as pd

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

train_out_path, val_out_path, test_out_path = [
    os.path.join(
        current_folder,
        '..',
        '..',
        'data',
        'amazon_query',
        f'Amazon_ESCI_Query_{i}_USLOCALE.json'
    ) for i in ['Train', 'Val', 'Test']
]

train = pd.read_json(train_path, lines=True)
val = pd.read_json(val_path, lines=True)
test = pd.read_json(test_path, lines=True)

#%%
print(f"len(train): {len(train)}, len(val): {len(val)}, len(test): {len(test)}")

#%%
train = train[train.product_locale == 'us']
val = val[val.product_locale == 'us']
test = test[test.product_locale == 'us']

print("After restricting to US locale")
print(f"len(train): {len(train)}, len(val): {len(val)}, len(test): {len(test)}")

# %%
train.to_json(train_out_path, orient='records', lines=True)
val.to_json(val_out_path, orient='records', lines=True)
test.to_json(test_out_path, orient='records', lines=True)