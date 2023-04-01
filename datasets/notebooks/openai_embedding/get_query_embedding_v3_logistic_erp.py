# %%
import pandas as pd
import json
from tqdm import tqdm
import numpy as np
from joblib import dump, load

# %%
df_train_emb = pd.read_json('/workspaces/multitask-llm-rnd/datasets/data/offshore_attr_extract/excel/erp/processed/offshore_amazon_erp_attr_train_oaiemb.json', 
    lines=True)
df_val_emb = pd.read_json('/workspaces/multitask-llm-rnd/datasets/data/offshore_attr_extract/excel/erp/processed/offshore_amazon_erp_attr_val_oaiemb.json', 
    lines=True)
df_test_emb = pd.read_json('/workspaces/multitask-llm-rnd/datasets/data/offshore_attr_extract/excel/erp/processed/offshore_amazon_erp_attr_test_oaiemb.json', 
    lines=True)

df_train_emb.loc[df_train_emb['attr_name_value_pairs_normalized_text'].apply(lambda x: len(x) == 0), 'attr_name_value_pairs_normalized_text'] = 'unknown'
df_val_emb.loc[df_val_emb['attr_name_value_pairs_normalized_text'].apply(lambda x: len(x) == 0), 'attr_name_value_pairs_normalized_text'] = 'unknown'
df_test_emb.loc[df_test_emb['attr_name_value_pairs_normalized_text'].apply(lambda x: len(x) == 0), 'attr_name_value_pairs_normalized_text'] = 'unknown'

# %%
X_train = np.array(df_train_emb['openai_embedding'].to_list())
X_val = np.array(df_val_emb['openai_embedding'].to_list())
X_test = np.array(df_test_emb['openai_embedding'].to_list())

# %%
print('X_train_val_test.shape', X_train.shape, X_val.shape, X_test.shape)

# %%
label2id = {}
with open('../../data/attribute_extraction_metadata_template/25L2_unfreetext_attribute_name_value_pairs_02232023.txt', 'r') as f:
    for l in f:
        i = l.replace('\n', '')
        if len(i) > 0:
            label2id[i] = len(label2id)
label2id['unknown'] = len(label2id)

# %%
len(label2id)

# %%
id2label = {label2id[i]: i for i in label2id}

# %%
y_train = np.zeros((len(X_train), len(label2id)))
y_val = np.zeros((len(X_val), len(label2id)))
y_test = np.zeros((len(X_test), len(label2id)))

# %%
for ind, i in enumerate(df_train_emb['attr_name_value_pairs_normalized_text'].to_list()):
    for j in i.split('\n'):
        if j in label2id:
            y_train[ind, label2id[j]] = 1

for ind, i in enumerate(df_val_emb['attr_name_value_pairs_normalized_text'].to_list()):
    for j in i.split('\n'):
        if j in label2id:
            y_val[ind, label2id[j]] = 1

for ind, i in enumerate(df_test_emb['attr_name_value_pairs_normalized_text'].to_list()):
    for j in i.split('\n'):
        if j in label2id:
            y_test[ind, label2id[j]] = 1

# %%
print('y_train_val_test.sum(1).mean()', y_train.sum(1).mean(), y_val.sum(1).mean(), y_test.sum(1).mean())

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (classification_report, accuracy_score, 
    label_ranking_average_precision_score)

# %%
clf = OneVsRestClassifier(estimator=LogisticRegression(), n_jobs=100, verbose=3)
clf.fit(X_train, y_train) 


# %%
dump(clf, 'simple_models/query_attrkv_clf_oaiemb_logistic_v3_erp.joblib')
