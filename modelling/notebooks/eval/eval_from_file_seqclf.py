# %%
import pandas as pd
import numpy as np
import dvc.api 
import json
from sklearn import metrics
import argparse
import os
import torch
from torch import nn
import logging
logging.getLogger().setLevel(logging.INFO)

from eval_from_file import (perf_eval_util, get_df_outfile, 
    read_json_helper, save_eval_results)

def perf_eval_utils_seqclf(df_pred, df_data, label_map_file, version, decode_method):
    logits = np.array(df_pred.logits.to_list())
    label_map = {}
    with open(label_map_file, 'r') as f:
        for l in f:
            l = l.replace('\n', '').strip()
            if len(l):
                label_map[l] = len(label_map)
    assert len(label_map) == logits.shape[1], (f"number of labels in label_map {len(label_map)} is different "
        f"from number of labels in logits {logits.shape[1]}")
    
    label_map_rev = {label_map[i]: i for i in label_map}

    df_tax = pd.read_json(
        dvc.api.get_url(
            'datasets/data/taxonomy/wish_newtax.json',
            repo='git@github.com:ContextLogic/multitask-llm-rnd.git'
        ), lines=True
    )
    df_tax['category_path'] = df_tax['category_path'].str.lower().str.strip()
    df_tax = df_tax[df_tax['category_path'] != '']
    df_tax['category_index'] = df_tax['category_path'].apply(lambda x: label_map[x])

    if decode_method == 'leaf':
        df_tax_leaf = df_tax[df_tax.is_leaf]
        leaf_index = df_tax_leaf['category_index'].to_numpy()
        predicted_index = leaf_index[logits[:, leaf_index].argmax(1)]
        predicted_category = [label_map_rev[i] for i in predicted_index]
        df_data[f'{version}_predicted_category'] = [i.split(" > ") for i in predicted_category]
    else:
        raise NotImplemented()
    return df_data
    

def main(args):
    """
    python notebooks/eval/eval_from_file_seqclf.py --dataset offshore-validated \
        --input /workspaces/multitask-llm-rnd/modelling/models/product_title_multitask_multimodal/version_1/seqclf-epoch=0-step=75000--wish_offshore_validated--test.json \
        --version mm_seqclf_redo_v0.1 \
        --label_map_file datasets/taxonomy/wish_v1.2.1_newtax_allpaths.txt \
        --decode_method leaf

    Save metrics to notebooks/eval/*.xlsx
    """
    df_data, output_file = get_df_outfile(args.dataset)
    
    output_file_aggregate = os.path.join(os.path.dirname(__file__), 'eval_aggregate.csv')
    infer_path = args.input
    version = args.version

    if version == 'lance':
        perm = pd.concat([perf_eval_util(df_data, level=i, col='lance_predicted_category') for i in [1, 2, 0, -1, -2]])
        df_pred = df_data[['title', 'category', 'lance_predicted_category']]
    else:
        df_pred = read_json_helper(infer_path)
        df_pred = df_pred.sort_values('batch_indices')
        assert len(df_data) == len(df_pred), f"df_data {len(df_data)} df_pred {len(df_pred)}"
        df_data = perf_eval_utils_seqclf(df_pred, df_data, args.label_map_file, version, args.decode_method)
        perm = pd.concat([perf_eval_util(df_data, level=i, col=f'{version}_predicted_category') for i in [1, 2, 0, -1, -2]])
        df_pred = df_data[['title', 'category', f'{version}_predicted_category']]
    
    perm['model_version'] = version
    save_eval_results(perm, output_file, df_pred, output_file_aggregate, version, args.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='mturk|offshore|offshore-validated')
    parser.add_argument('--input', help='input inference file used')
    parser.add_argument('--version', help='model version used, "lance" being lance\'s model')
    parser.add_argument('--label_map_file', help='label file')
    parser.add_argument('--decode_method', help='leaf|top-down-greedy|brute-force')

    args = parser.parse_args()
    assert args.dataset in ['mturk', 'offshore', 'offshore-validated']
    main(args)