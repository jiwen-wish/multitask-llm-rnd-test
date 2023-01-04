#%%
import os
import pandas as pd
import dvc.api


params = dvc.api.params_show()

sample_amount = params['sample_wish_queries_offshore']['amount']
sample_head_ratio = params['sample_wish_queries_offshore']['head_ratio']

df_queries = pd.read_json(os.path.join(
    os.path.dirname(__file__),
    'top_used_wish_queries.json'
), lines=True).sort_values('cnt', ascending=False)

# %%
head_amount = int(sample_head_ratio * sample_amount)
tail_amount = sample_amount - head_amount
df_head = df_queries.head(head_amount).reset_index()
df_tail = df_queries.tail(len(df_queries) - head_amount).sample(tail_amount,
    random_state=42, replace=False).reset_index()
df_head['sample_method'] = 'head'
df_tail['sample_method'] = 'uniform'

# %%
df_sample = pd.concat([df_head, df_tail]).sample(frac=1.0 ,random_state=42)

# %%
df_sample.to_json(os.path.join(
    os.path.dirname(__file__),
    'top_used_wish_queries_offshore_sample_100000.json'
), orient='records', lines=True)

df_sample.to_excel(os.path.join(
    os.path.dirname(__file__),
    'top_used_wish_queries_offshore_sample_100000.xlsx'
), index=False)