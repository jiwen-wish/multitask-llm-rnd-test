#%%
import pandas as pd 
import gzip 
import os 
import json
from tqdm import tqdm
#%%
curdir = os.path.dirname(__file__)
tahoe_dedup_train_path, tahoe_dedup_val_path, tahoe_dedup_test_path = [
    os.path.join(curdir, '..', '..', 'data', 'wish_products_truetag_tahoe', f'Wish_Truetag_Tahoe_Meta_{i}_Dedup.json') \
        for i in ['Train', 'Val', 'Test']
]
clipmore_gzip_path = os.path.join(curdir, '..', '..', 'data', 'wish_clipmore', 'wishproducts_clipmore_tahoe.json.gz')

clipmore_out_train_json_path, clipmore_out_val_json_path, clipmore_out_test_json_path, clipmore_out_rest_json_path = [
    os.path.join(curdir, '..', '..', 'data', 'wish_clipmore', f'Wish_Clipmore_Tahoe_{i}_Dedup.json') \
        for i in ['Train', 'Val', 'Test', 'Rest']
]

df_newtax = pd.read_json(os.path.join(curdir, '..', '..', 'data', 'taxonomy', 'wish_newtax.json'), lines=True)

id2newtax = {}
for i in df_newtax.to_dict('records'):
    id2newtax[str(i['id'])] = i['category_path'].lower().strip()

#%%
uniq_train_titles = set()
uniq_val_titles = set()
uniq_test_titles = set()

with open(tahoe_dedup_train_path, 'r') as f:
    for l in tqdm(f):
        if len(l) > 0:
            uniq_train_titles.add(json.loads(l)['title'])
with open(tahoe_dedup_val_path, 'r') as f:
    for l in tqdm(f):
        if len(l) > 0:
            uniq_val_titles.add(json.loads(l)['title'])
with open(tahoe_dedup_test_path, 'r') as f:
    for l in tqdm(f):
        if len(l) > 0:
            uniq_test_titles.add(json.loads(l)['title'])
            
# %%
from collections import defaultdict
clip_titles = {}
train_c, val_c, test_c = 0, 0, 0
rest_c = 0
truetag_c, v121_c = 0, 0
with gzip.open(clipmore_gzip_path, 'r') as f, \
    open(clipmore_out_train_json_path, 'w') as f_train, \
    open(clipmore_out_val_json_path, 'w') as f_val, \
    open(clipmore_out_test_json_path, 'w') as f_test, \
    open(clipmore_out_rest_json_path, 'w') as f_rest:
    for l in tqdm(f):
        if len(l):
            try:
                dat = json.loads(l)
                dat['truetag_categories'] = dat['categories']
                del dat['categories']
                # take first
                dat['truetag_category'] = dat['truetag_categories'][0] if len(dat['truetag_categories']) else []
                try:
                    dat['v121_category'] = id2newtax[dat["category_id_path"].split(",")[-1]].split(" > ")
                except:
                    dat['v121_category'] = []
                del dat["category_id_path"]
                del dat["true_tags_are_predicted"]
                del dat["true_tag_ids"]
                if len(dat['truetag_category']) > 0:
                    truetag_c += 1
                if len(dat['v121_category']) > 0:
                    v121_c += 1
                title = dat['title']
                if title in clip_titles:
                    continue 
                else:
                    if title in uniq_train_titles:
                        f_train.write(json.dumps(dat) + '\n')
                        train_c += 1
                    elif title in uniq_val_titles:
                        f_val.write(json.dumps(dat) + '\n')
                        val_c += 1
                    elif title in uniq_test_titles:
                        f_test.write(json.dumps(dat) + '\n')
                        test_c += 1
                    else:
                        f_rest.write(json.dumps(dat) + '\n')
                        rest_c += 1
                    clip_titles[title] = 1
            except Exception as e:
                print(e)

print(f"train_c: {train_c}, val_c: {val_c}, test_c: {test_c}, rest_c: {rest_c}, truetag_c: {truetag_c}, v121_c: {v121_c}")
# %%
