#%%
import numpy as np 
np.random.seed(42)
import os 
from tqdm import tqdm
import json

curdir = os.path.dirname(__file__)
clipmore_train_json_joinpseudo_path = os.path.join(
    curdir, '..', '..', 'data', 'wish_clipmore', 'Wish_Clipmore_Tahoe_Train_Dedup_Clean_JoinPseudo.json')
resplit_train_path, resplit_val_path, resplit_test_path = [
    os.path.join(curdir, '..', '..', 'data', 'wish_clipmore', f'Wish_Clipmore_Tahoe_Train_Dedup_Clean_JoinPseudo_Resplit_{i}.json') \
        for i in ['Train', 'Val', 'Test']
]

with open(clipmore_train_json_joinpseudo_path, "r") as f_in, \
        open(resplit_train_path, "w") as f_train, \
        open(resplit_val_path, "w") as f_val, \
        open(resplit_test_path, "w") as f_test:
    for l in tqdm(f_in):
        if len(l):
            dat = json.loads(l)
            dat["v121_category_multitaskv0_1_pseudo"] = dat["v121_category_multitaskv0.1_pseudo"]
            del dat["v121_category_multitaskv0.1_pseudo"]
            rand = np.random.random()
            l_ = json.dumps(dat) + "\n"
            if rand < 0.001:
                f_test.write(l_)
            elif rand < 0.005:
                f_val.write(l_)
            else:
                f_train.write(l_)

# %%
