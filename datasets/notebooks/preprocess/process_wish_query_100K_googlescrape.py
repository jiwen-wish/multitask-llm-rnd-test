#%%
import os
import pandas as pd
import numpy as np

dirname = os.path.dirname(__file__)
scrape_file = os.path.join(dirname, "..", "..", "data", "query", "top_used_wish_queries_offshore_sample_100000_scraped_googlesearch.json")
df_scrape = pd.read_json(scrape_file, lines=True)
train_path, val_path, test_path = [os.path.join(dirname, "..", "..", "data", "query", 
    f"top_used_wish_queries_offshore_sample_100000_scraped_googlesearch_{mode}.json") for mode in [
        "Train", "Val", "Test"]]
#%%
df_scrape_clean = df_scrape[(~df_scrape.title.apply(lambda x: x.endswith("..."))) & \
    (df_scrape.title.apply(lambda x: len(x) > 0))].drop_duplicates(["query", "title"])
# %%
query_list = sorted(list(set(df_scrape_clean['query'])))
np.random.seed(42)
np.random.shuffle(query_list)

train_query = set(query_list[:int(len(query_list) * .8)])
val_query = set(query_list[int(len(query_list) * .8):int(len(query_list) * .9)])
test_query = set(query_list[int(len(query_list) * .9):])

print(f"train_query: {len(train_query)}, val_query: {len(val_query)}, test_query: {len(test_query)}")

df_train = df_scrape_clean[df_scrape_clean['query'].apply(lambda x: x in train_query)]
df_val = df_scrape_clean[df_scrape_clean['query'].apply(lambda x: x in val_query)]
df_test = df_scrape_clean[df_scrape_clean['query'].apply(lambda x: x in test_query)]


# %%
df_train.to_json(train_path, orient='records', lines=True)
df_val.to_json(val_path, orient='records', lines=True)
df_test.to_json(test_path, orient='records', lines=True)