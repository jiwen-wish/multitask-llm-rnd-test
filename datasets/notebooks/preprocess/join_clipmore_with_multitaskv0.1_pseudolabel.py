#%%
import pandas as pd
import dvc.api 
from tqdm import tqdm
import os 
import json
import numpy as np

curdir = os.path.dirname(__file__)
resource_url = dvc.api.get_url(
    'datasets/wish_tahoe_dedup_pseudolabel/wish_tahoe_title_dedup_multitask_v0.1_preds_allmatch.json.gz',
    repo='git@github.com:junwang-wish/query_understanding_model.git'
)

clipmore_train_json_path, clipmore_val_json_path, clipmore_test_json_path = [
    os.path.join(curdir, '..', '..', 'data', 'wish_clipmore', f'Wish_Clipmore_Tahoe_{i}_Dedup_Clean.json') \
        for i in ['Train', 'Val', 'Test']
]

clipmore_rest_path_unclean = os.path.join(curdir, '..', '..', 'data', 'wish_clipmore', 'Wish_Clipmore_Tahoe_Rest_Dedup.json')

clipmore_train_json_joinpseudo_path, clipmore_val_json_joinpseudo_path, clipmore_test_json_joinpseudo_path = [
    os.path.join(curdir, '..', '..', 'data', 'wish_clipmore', f'Wish_Clipmore_Tahoe_{i}_Dedup_Clean_JoinPseudo.json') \
        for i in ['Train', 'Val', 'Test']
]

clipmore_val_json_joinpseudo_path_unclean, clipmore_test_json_joinpseudo_path_unclean = [
    os.path.join(curdir, '..', '..', 'data', 'wish_clipmore', f'Wish_Clipmore_Tahoe_Rest_Dedup_Unclean_JoinPseudo_{i}.json') \
        for i in ['Val', 'Test']
]

#%%
chunks = pd.read_json(resource_url, lines=True, chunksize=1000, compression='gzip')
# %%
title2pseudo_idx = {}
pesudotuple2idx = {}
for df_chunk in tqdm(chunks):
    for i in df_chunk[['title', 'multitask_seqclf_v0.1_predicted_category']].to_dict('records'):
        pseudotuple = tuple(i['multitask_seqclf_v0.1_predicted_category'])
        title = i['title']
        if pseudotuple not in pesudotuple2idx:
            pesudotuple2idx[pseudotuple] = len(pesudotuple2idx)
        title2pseudo_idx[title] = pesudotuple2idx[pseudotuple]

pesudotuple2idx_rev = {pesudotuple2idx[i]: i for i in pesudotuple2idx}

seen_titles = {}

with open(clipmore_train_json_joinpseudo_path, 'w') as f_out, open(clipmore_train_json_path, 'r') as f_in:
    for l in tqdm(f_in):
        if len(l):
            dat = json.loads(l)
            title = dat['title']
            if title in title2pseudo_idx:
                dat['v121_category_multitaskv0.1_pseudo'] = list(pesudotuple2idx_rev[title2pseudo_idx[title]])
                seen_titles[title] = 1
                f_out.write(json.dumps(dat) + "\n")

with open(clipmore_val_json_joinpseudo_path, 'w') as f_out, open(clipmore_val_json_path, 'r') as f_in:
    for l in tqdm(f_in):
        if len(l):
            dat = json.loads(l)
            title = dat['title']
            if title in title2pseudo_idx:
                dat['v121_category_multitaskv0.1_pseudo'] = list(pesudotuple2idx_rev[title2pseudo_idx[title]])
                seen_titles[title] = 1
                f_out.write(json.dumps(dat) + "\n")

with open(clipmore_test_json_joinpseudo_path, 'w') as f_out, open(clipmore_test_json_path, 'r') as f_in:
    for l in tqdm(f_in):
        if len(l):
            dat = json.loads(l)
            title = dat['title']
            if title in title2pseudo_idx:
                dat['v121_category_multitaskv0.1_pseudo'] = list(pesudotuple2idx_rev[title2pseudo_idx[title]])
                seen_titles[title] = 1
                f_out.write(json.dumps(dat) + "\n")

with open(clipmore_val_json_joinpseudo_path_unclean, 'w') as f_out_val, \
        open(clipmore_test_json_joinpseudo_path_unclean, 'w') as f_out_test, \
            open(clipmore_rest_path_unclean, 'r') as f_in:
    for l in tqdm(f_in):
        if len(l):
            dat = json.loads(l)
            title = dat['title']
            if title not in seen_titles:
                if title in title2pseudo_idx:
                    dat['v121_category_multitaskv0.1_pseudo'] = list(pesudotuple2idx_rev[title2pseudo_idx[title]])
                    if np.random.random() < .8:
                        f_out_val.write(json.dumps(dat) + "\n")
                    else:
                        f_out_test.write(json.dumps(dat) + "\n")
