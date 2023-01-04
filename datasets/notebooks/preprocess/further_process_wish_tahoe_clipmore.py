#%%
import pandas as pd 
import gzip 
import os 
import json
from tqdm import tqdm
#%%
curdir = os.path.dirname(__file__)

clipmore_out_train_json_path, clipmore_out_val_json_path, clipmore_out_test_json_path, clipmore_out_rest_json_path = [
    os.path.join(curdir, '..', '..', 'data', 'wish_clipmore', f'Wish_Clipmore_Tahoe_{i}_Dedup.json') \
        for i in ['Train', 'Val', 'Test', 'Rest']
]

clipmore_out_train_json_path_clean, clipmore_out_val_json_path_clean, clipmore_out_test_json_path_clean = [
    os.path.join(curdir, '..', '..', 'data', 'wish_clipmore', f'Wish_Clipmore_Tahoe_{i}_Dedup_Clean.json') \
        for i in ['Train', 'Val', 'Test']
]

train_c, val_c, test_c, rest_c = 0, 0, 0, 0
train_c_clean, val_c_clean, test_c_clean = 0, 0, 0


with open(clipmore_out_train_json_path, 'r') as f_train, \
    open(clipmore_out_val_json_path, 'r') as f_val, \
    open(clipmore_out_test_json_path, 'r') as f_test, \
    open(clipmore_out_rest_json_path, 'r') as f_rest, \
    open(clipmore_out_train_json_path_clean, 'w', buffering=1) as f_train_c, \
    open(clipmore_out_val_json_path_clean, 'w', buffering=1) as f_val_c, \
    open(clipmore_out_test_json_path_clean, 'w', buffering=1) as f_test_c:
    for l in tqdm(f_train):
        if len(l):
            train_c += 1
            dat = json.loads(l)
            if len(dat['truetag_category']) > 0 and len(dat['v121_category']) > 0:
                if isinstance(dat["img_embedding"], str):
                    dat["img_embedding"] = eval(dat["img_embedding"])
                if len(dat["img_embedding"]) == 768:
                    f_train_c.write(json.dumps(dat) + "\n")
                    train_c_clean += 1
                else:
                    print("bad img_embedding", len(dat["img_embedding"]))
    for l in tqdm(f_val):
        if len(l):
            val_c += 1
            dat = json.loads(l)
            if len(dat['truetag_category']) > 0 and len(dat['v121_category']) > 0:
                if isinstance(dat["img_embedding"], str):
                    dat["img_embedding"] = eval(dat["img_embedding"])
                if len(dat["img_embedding"]) == 768:
                    f_val_c.write(json.dumps(dat) + "\n")
                    val_c_clean += 1
                else:
                    print("bad img_embedding", len(dat["img_embedding"]))
    for l in tqdm(f_test):
        if len(l):
            test_c += 1
            dat = json.loads(l)
            if len(dat['truetag_category']) > 0 and len(dat['v121_category']) > 0:
                if isinstance(dat["img_embedding"], str):
                    dat["img_embedding"] = eval(dat["img_embedding"])
                if len(dat["img_embedding"]) == 768:
                    f_test_c.write(json.dumps(dat) + "\n")
                    test_c_clean += 1
                else:
                    print("bad img_embedding", len(dat["img_embedding"]))
    for l in tqdm(f_rest):
        if len(l):
            rest_c += 1
            dat = json.loads(l)
            if len(dat['truetag_category']) > 0 and len(dat['v121_category']) > 0:
                if isinstance(dat["img_embedding"], str):
                    dat["img_embedding"] = eval(dat["img_embedding"])
                if len(dat["img_embedding"]) == 768:
                    f_val_c.write(json.dumps(dat) + "\n")
                    val_c_clean += 1
                else:
                    print("bad img_embedding", len(dat["img_embedding"]))
            
print(f"train_c: {train_c}, val_c: {val_c}, test_c: {test_c}, rest_c: {rest_c}, train_c_clean: {train_c_clean}, val_c_clean: {val_c_clean}, test_c_clean: {test_c_clean}")
# %%
