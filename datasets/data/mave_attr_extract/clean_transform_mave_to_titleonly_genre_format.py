# %%
import json 
import pandas as pd
from tqdm import tqdm
import os 
from collections import defaultdict



dirpath = os.path.dirname(__file__)
mave_pos_input_path = os.path.join(dirpath, 'mave_positives.jsonl')
mave_pos_output_path = os.path.join(dirpath, 'mave_positives_titleonly_genreformat.json')
mave_neg_input_path = os.path.join(dirpath, 'mave_negatives.jsonl')
mave_neg_output_path = os.path.join(dirpath, 'mave_negatives_titleonly_genreformat.json')

# %%
recs = []
with open(mave_pos_input_path, 'r') as f:
    for l in tqdm(f):
        recs.append(json.loads(l))
        if len(recs) <0:
            break 
df_pos = pd.DataFrame(recs)

recs = []
with open(mave_neg_input_path, 'r') as f:
    for l in tqdm(f):
        recs.append(json.loads(l))
        if len(recs) <0:
            break 
df_neg = pd.DataFrame(recs)


# %%
print(f'len(df_pos): {len(df_pos)}, len(df_neg): {len(df_neg)}')

# %%
df_pos['attr_vals'] = df_pos['attributes'].apply(lambda x: [(i['key'].strip().lower(), j['value'].strip().lower()) for i in x for j in i['evidences']])

# %%
df_neg['attr_vals'] = df_neg['attributes'].apply(lambda x: [(i['key'].strip().lower(),j.strip().lower()) for i in x for j in (i['evidences'] + [""])])

# %%
from collections import defaultdict

# %%
cat_attr_val_cnt_pos = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

# %%
for i in df_pos.to_dict('records'):
    for j in i['attr_vals']:
        cat_attr_val_cnt_pos[i['category'].lower().strip()][j[0]][j[1]] += 1

# %%
cat_attr_val_cnt_neg = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

# %%
for i in df_neg.to_dict('records'):
    for j in i['attr_vals']:
        cat_attr_val_cnt_neg[i['category'].lower().strip()][j[0]][j[1]] += 1

# %%
print("#pos_uniq_cat: {}, #pos_uniq_attr: {}, #pos_uniq_cat_attr_pair: {}, #pos_uniq_cat_attr_val_pair: {}".format(
    len(set([(i,) for i in cat_attr_val_cnt_pos for j in cat_attr_val_cnt_pos[i] for k in cat_attr_val_cnt_pos[i][j]])), 
    len(set([(j,) for i in cat_attr_val_cnt_pos for j in cat_attr_val_cnt_pos[i] for k in cat_attr_val_cnt_pos[i][j]])),
    len(set([(i,j) for i in cat_attr_val_cnt_pos for j in cat_attr_val_cnt_pos[i] for k in cat_attr_val_cnt_pos[i][j]])),
    len(set([(i,j,k) for i in cat_attr_val_cnt_pos for j in cat_attr_val_cnt_pos[i] for k in cat_attr_val_cnt_pos[i][j]]))
))

print("#neg_uniq_cat: {}, #neg_uniq_attr: {}, #neg_uniq_cat_attr_pair: {}, #neg_uniq_cat_attr_val_pair: {}".format(
    len(set([(i,) for i in cat_attr_val_cnt_neg for j in cat_attr_val_cnt_neg[i] for k in cat_attr_val_cnt_neg[i][j]])), 
    len(set([(j,) for i in cat_attr_val_cnt_neg for j in cat_attr_val_cnt_neg[i] for k in cat_attr_val_cnt_neg[i][j]])),
    len(set([(i,j) for i in cat_attr_val_cnt_neg for j in cat_attr_val_cnt_neg[i] for k in cat_attr_val_cnt_neg[i][j]])),
    len(set([(i,j,k) for i in cat_attr_val_cnt_neg for j in cat_attr_val_cnt_neg[i] for k in cat_attr_val_cnt_neg[i][j]]))
))
# %%
match_errors = []
match_success = []
match_sources = []
recs = []
for i in tqdm(df_pos.to_dict('records')):
    has_error = False
    for a in i['attributes']:
        for v in a['evidences']:
            qa_model_extracted_text = i['paragraphs'][v['pid']]['text'][v['begin']: v['end']]
            normalized_value = v['value']
            match_sources.append((i['id'], i['category'], a['key'], v['value'], i['paragraphs'][v['pid']]['source'], normalized_value))
            if normalized_value != qa_model_extracted_text:
                match_errors.append((i['id'], i['category'], a['key'], v['pid'], normalized_value, qa_model_extracted_text))
                has_error = True
            else:
                match_success.append((i['id'], i['category'], a['key'], v['pid'], normalized_value, qa_model_extracted_text))
    if not has_error:
        recs.append(i)
df_pos_correct = pd.DataFrame(recs)

# %%
print("df_pos attr span mismatch error rate: {}".format(len(match_errors) / (len(match_success) + len(match_errors))))

# %%
print("len(df_pos_correct) / len(df_pos): {}".format(len(df_pos_correct) / len(df_pos)))

# %%
from collections import Counter

# %%
df_match_sources = pd.DataFrame(match_sources, columns=['id', 'cat', 'attr', 'val', 'source', 'evidence'])

# %%
tmp = df_match_sources.groupby('id').agg({
    'source': lambda x: [i for i in x],
    'val': lambda x: [i for i in x]
}).reset_index()

# %%
recs = []
for i in tqdm(tmp.to_dict('records')):
    uniq_vals = {}
    needed_s_v = []
    for ind, (s, v) in enumerate(zip(i['source'], i['val'])):
        v = v.lower().strip()
        if v not in uniq_vals:
            needed_s_v.append((ind, s, v))
            uniq_vals[v] = 1
    i['needed_source_value'] = needed_s_v
    recs.append(i)
tmp2 = pd.DataFrame(recs)

# %%
tmp2['needed_source'] = tmp2.needed_source_value.apply(lambda x: tuple(set([i[1] for i in x])))

# %%
tmp = Counter(tmp2['needed_source'])

# %%
import numpy as np


# %%
tmp = tmp2[tmp2.needed_source != ('title',)]
tmp['title_ratio'] = tmp['needed_source_value'].apply(lambda x: np.mean([i[1] == 'title' for i in x]))

# %%
tmp2['title_ratio'] = tmp2['needed_source_value'].apply(lambda x: np.mean([i[1] == 'title' for i in x]))

# %%
df_pos_correct = tmp2[['id', 'needed_source_value', 'title_ratio']].merge(df_pos_correct, on='id', how='inner')

# %%
df_pos_correct_title = df_pos_correct[df_pos_correct.title_ratio >= 0.5]

# %%
df_pos_correct_title['title'] = df_pos_correct_title['paragraphs'].apply(lambda x: x[0]['text'] if x[0]['source'] == 'title' else None)

# %%
recs = []
for i in tqdm(df_pos_correct_title.to_dict('records')):
    title_attributes = {a['key']: [j for j in a['evidences'] if i['paragraphs'][j['pid']]['source'] == 'title'] for a in i['attributes']}
    title_attributes = {j: title_attributes[j] for j in title_attributes if len(title_attributes[j]) > 0}
    i['title_attributes'] = title_attributes
    recs.append(i)

# %%
df_pos_correct_title = pd.DataFrame(recs)

# %%
df_pos_correct_title_simple = df_pos_correct_title[['id', 'title', 'category', 'title_attributes']].drop_duplicates('title')

# %%
df_pos_correct_title_simple[df_pos_correct_title_simple.title.apply(lambda x: '](' in x)]

# %%
print("len(df_pos_correct_title_simple) / len(df_pos): {}".format(len(df_pos_correct_title_simple) / len(df_pos)))

# %%
recs = []
tmp = df_pos_correct_title_simple.to_dict('records')

# %%

for i in tqdm(tmp):
    t_i = i['title'].replace('[', " ").replace(']', " ").replace('(', " ").replace(')', " ")
    if '->' in t_i:
        continue
    t_o = t_i
    attr_val_evi_dicts = [({**k, **{'attribute': j}}) for j in i['title_attributes'] for k in i['title_attributes'][j]]
    begin_ends = {}
    inds = []
    discard_sample = False
    for ind, d in enumerate(attr_val_evi_dicts):
        b_, e_ = d['begin'], d['end']
        if len(begin_ends) == 0:
            begin_ends[(b_, e_)] = 1
            inds.append(ind)
        else:
            not_intersect = True
            for (b, e) in begin_ends:
                if b_ <= b < e_ or b_ < e <= e_ or b <= b_ < e or b < e_ <= e:
                    # print((b, e), 'intersects', (b_, e_), i, 'discarded')
                    discard_sample = True 
                    break
            begin_ends[(b_, e_)] = 1
        if discard_sample:
            break 
    if discard_sample:
        # print('-' * 20)
        continue
        
    for d in sorted(attr_val_evi_dicts, key=lambda x: -x['end']):
        t_o = t_o[:d['begin']] + '['+ t_o[d['begin']:d['end']] + ']' + '(' + d['attribute'] + ')' + t_o[d['end']:]
    t_i_c = t_i.strip()
    t_o_c = t_o.strip()
    while True:
        t_i_c_ = t_i_c.replace("  ", " ")
        t_o_c_ = t_o_c.replace("  ", " ")
        if t_i_c_ == t_i_c and t_o_c_ == t_o_c:
            break
        else:
            t_i_c = t_i_c_ 
            t_o_c = t_o_c_
        
    i['text'] = t_i_c + " -> " + t_o_c
    recs.append(i)
df_pos_correct_title_simple_aug = pd.DataFrame(recs)

# %%
print("len(df_pos_correct_title_simple_aug) / len(df_pos): {}".format(len(df_pos_correct_title_simple_aug) / len(df_pos)))

# %%
df_pos_correct_title_simple_aug = df_pos_correct_title_simple_aug.rename(columns={'category': 'category_flat'})

# %%
def remove_special_char(x):
    x = x.replace('[', " ").replace(']', " ").replace('(', " ").replace(')', " ")
    x = x.strip()
    while True:
        x_ = x.replace("  ", " ")
        if x == x_:
            return x 
        else:
            x = x_

df_neg_aug =  df_neg
df_neg_aug = df_neg_aug.rename(columns={'category': 'category_flat'})
df_neg_aug['title'] = df_neg_aug['paragraphs'].apply(lambda x: x[0]['text'] if x[0]['source'] == 'title' else None)
df_neg_aug = df_neg_aug[df_neg_aug.title.apply(lambda x: '->' not in x)]
tmp = df_neg_aug['title'].apply(remove_special_char)
df_neg_aug['text'] = tmp + " -> " + tmp
df_neg_aug['title_attributes'] = df_neg_aug['attributes'].apply(lambda x: {i['key']: [] for i in x})
df_neg_aug = df_neg_aug[['id', 'title', 'category_flat','title_attributes', 'text']]

# %%
print("len(df_neg_aug) / len(df_neg): {}".format(len(df_neg_aug) / len(df_neg)))

# %%
df_pos_correct_title_simple_aug.to_json(mave_pos_output_path, orient='records', lines=True)
df_neg_aug.to_json(mave_neg_output_path, orient='records', lines=True)
# %%
print('write {} pos samples ({} percent of df_pos) to {} and {} neg samples ({} of df_neg) samples to {}'.format(
    len(df_pos_correct_title_simple_aug), 
    len(df_pos_correct_title_simple_aug) / len(df_pos) * 100,
    mave_pos_output_path, 
    len(df_neg_aug), 
    len(df_neg_aug) / len(df_neg) * 100, 
    mave_neg_output_path
))



