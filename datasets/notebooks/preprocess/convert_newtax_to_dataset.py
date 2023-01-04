#%%
import pandas as pd 
import os 

df_tax = pd.read_json(
    os.path.join(
        os.path.dirname(__file__),
        '..',
        '..',
        'data',
        'taxonomy',
        'wish_newtax.json'
    ), lines=True
)

out_path = os.path.join(
    os.path.dirname(__file__),
    '..',
    '..',
    'data',
    'taxonomy',
    'wish_newtax_converted_to_data.json'
)

#%%
df_tax['title'] = df_tax['category_path'].str.strip().str.lower()
df_tax['category'] = df_tax['title'].apply(lambda x: x.split(" > "))
df_tax['text'] = df_tax['title'] + " -> " + df_tax['category'].apply(lambda x: "".join(["[" + i + "]" for i in x]))
# %%
df_tax = df_tax[df_tax['title'] != '']
# %%
df_tax = df_tax[['title', 'category', 'text', 'is_leaf']]
# %%
df_tax.to_json(out_path, orient='records', lines=True)