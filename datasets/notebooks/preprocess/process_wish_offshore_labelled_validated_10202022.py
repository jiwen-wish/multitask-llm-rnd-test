#%%
import pandas as pd 
import os 

df_excel = pd.read_excel(
    os.path.join(
        os.path.dirname(__file__),
        '..',
        '..',
        'data',
        'wish_products_internallabel',
        'Internal-Label-Validation-10202022.xlsx'
    ), sheet_name=None
)['Sample (50L1)']

df_tax = pd.read_json(
    os.path.join(
        os.path.dirname(__file__),
        '..',
        '..',
        'data',
        'taxonomy',
        'wish_newtax.json'
    ),lines=True
)

out_path = os.path.join(
    os.path.dirname(__file__),
    '..',
    '..',
    'data',
    'wish_products_internallabel',
    'wish_products_offshore_labelled_validated_processed.json'
)

#%%
df_excel_clean = df_excel[df_excel['Labeled Leaf Correct?']]
df_excel_clean = df_excel_clean[['pid', 'title', 'labeler_path', 'pred_path']]
assert set(df_excel_clean['labeler_path']).issubset(set(df_tax.category_path))
assert set(df_excel_clean['pred_path']).issubset(set(df_tax.category_path))
df_excel_clean['lance_predicted_category'] = df_excel_clean['pred_path'].apply(lambda x: x.lower().strip().split(" > "))
df_excel_clean['category'] = df_excel_clean['labeler_path'].apply(lambda x: x.lower().strip().split(" > "))
df_excel_clean['text'] = df_excel_clean['title'] + " -> " + df_excel_clean['category'].apply(
    lambda x: ''.join(['[' + i + ']' for i in x]))
df_excel_clean = df_excel_clean[['pid', 'title', 'text', 'category', 'lance_predicted_category']]
# %%
df_excel_clean.to_json(out_path, lines=True, orient='records')