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

def perf_eval_util_emb(df_pred, df_pred_tax, df_data, version, distance_func):
    emb_title = np.array(df_pred.embedding.to_list())
    emb_tax = np.array(df_pred_tax.embedding.to_list())
    
    if distance_func == 'cosine':
        try:
            sims = nn.functional.normalize(torch.tensor(emb_title).cuda(), 1).mm(
                nn.functional.normalize(torch.tensor(emb_tax).cuda(), 1).T).detach().cpu().numpy()
        except Exception as e:
            logging.warning(f"cuda eval failed with {e}, try with cpu")
            sims = nn.functional.normalize(torch.tensor(emb_title), 1).mm(
                nn.functional.normalize(torch.tensor(emb_tax), 1).T).detach().cpu().numpy()
    elif distance_func == 'order':
        try:
            sims = -(torch.pow(
                    torch.clamp(
                        nn.functional.normalize(torch.tensor(emb_tax).cuda(), 1).unsqueeze(1) - \
                            nn.functional.normalize(torch.tensor(emb_title).cuda(), 1), min=0
                    ), 2
                ).sum(2)).detach().cpu().numpy().T
        except Exception as e:
            logging.warning(f"cuda eval failed with {e}, try with cpu")
            sims = -(torch.pow(
                    torch.clamp(
                        nn.functional.normalize(torch.tensor(emb_tax), 1).unsqueeze(1) - \
                            nn.functional.normalize(torch.tensor(emb_title), 1), min=0
                    ), 2
                ).sum(2)).detach().cpu().numpy().T
    else:
        raise NotImplemented()

    df_taxname = pd.read_json(
        dvc.api.get_url(
            'data/taxonomy/wish_newtax_converted_to_data.json',
            repo='git@github.com:ContextLogic/multitask-llm-rnd.git'
        ), lines=True
    )

    df_taxname_leaf = df_taxname[df_taxname.is_leaf].reset_index()
    df_data[f'{version}_predicted_category'] = df_taxname_leaf.loc[
        sims[:, df_taxname_leaf['index']].argmax(1)]['title'].apply(
            lambda x: x.split(' > ')).tolist()
    
    return df_data

def main(args):
    """
    python notebooks/eval/eval_from_file_emb.py --dataset offshore-validated \
        --input_title models/product_title_multitask_multimodal/version_1/emb-epoch=0-step=75000--wish_offshore_validated_wclip--multimodal--inputemb.json \
        --input_tax models/product_title_multitask_multimodal/version_1/emb-epoch=0-step=75000--wish-newtax-v1.2.1--outputemb.json \
        --version mm_emb_clip_v0.1 \
        --distance_func cosine

    Save metrics to notebooks/eval/*.xlsx
    """
    df_data, output_file = get_df_outfile(args.dataset)
    
    output_file_aggregate = os.path.join(os.path.dirname(__file__), 'eval_aggregate.csv')
    infer_path_title = args.input_title
    infer_path_tax = args.input_tax
    version = args.version

    if version == 'lance':
        perm = pd.concat([perf_eval_util(df_data, level=i, col='lance_predicted_category') for i in [1, 2, 0, -1, -2]])
        df_pred = df_data[['title', 'category', 'lance_predicted_category']]
    else:
        df_pred = read_json_helper(infer_path_title) # title
        df_pred = df_pred.sort_values('batch_indices')
        assert len(df_data) == len(df_pred), f"df_data {len(df_data)} df_pred {len(df_pred)}"
        df_pred_tax = read_json_helper(infer_path_tax) # tax
        df_data = perf_eval_util_emb(df_pred, df_pred_tax, df_data, version, args.distance_func)
        perm = pd.concat([perf_eval_util(df_data, level=i, col=f'{version}_predicted_category') for i in [1, 2, 0, -1, -2]])
        df_pred = df_data[['title', 'category', f'{version}_predicted_category']]
    
    perm['model_version'] = version
    save_eval_results(perm, output_file, df_pred, output_file_aggregate, version, args.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='mturk|offshore|offshore-validated')
    parser.add_argument('--input_title', help='input inference file used for title')
    parser.add_argument('--input_tax', help='input inference file used for taxonomy')
    parser.add_argument('--version', help='model version used, "lance" being lance\'s model')
    parser.add_argument('--distance_func', help='distance func used for emb model')

    args = parser.parse_args()
    assert args.dataset in ['mturk', 'offshore', 'offshore-validated']
    main(args)