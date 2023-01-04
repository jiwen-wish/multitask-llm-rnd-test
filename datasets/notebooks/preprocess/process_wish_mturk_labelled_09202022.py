# %%
import pandas as pd
import os

current_folder = os.path.dirname(__file__)

wish_mturk_file_path = os.path.join(
    current_folder,
    '..',
    '..',
    'data',
    'wish_products',
    'wish-mturk-labelled-09202022.xlsx'
)

wish_mturk_test_path = os.path.join(
    current_folder,
    '..',
    '..',
    'data',
    'wish_products',
    'wish-mturk-labelled-09202022-clean.json'
)


# %%
df = pd.read_excel(wish_mturk_file_path)

# %%
df = df.rename(columns={'a1': 'category', 'Title': 'title'})

# %%
df.loc[df['Correct/Incorrect'] == 'Incorrect', 'category'] = df.loc[df['Correct/Incorrect'] == 'Incorrect', 'Right Category']

# %%
df = df[['pid', 'title', 'category']]

# %%
df['category'] = df['category'].apply(lambda x: [i.lower() for i in x.split(' > ')])
df['text'] = df.title.str.strip() + ' -> ' + \
    df.category.apply(lambda x: ''.join(['[' + i.lower().strip() + ']' for i in x]))

# %%
df.to_json(wish_mturk_test_path, orient='records', lines=True)
