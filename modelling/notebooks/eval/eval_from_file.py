# %%
import dvc.api 
import pandas as pd
import json
from sklearn import metrics
import argparse
import os

# %%
def perf_eval_util(df, level, col):
    """Evaluate performance at arbitrary forward or backward level 
    1: L1 cateogry (root level)
    2: L2 category
    3: L3 category
    ...
    0: Leaf category
    -1: Parent category
    -2: Grandparent category
    ...
    """
    assert isinstance(level, int)
    labeler = 'category'
    model = col
    if level > 0:
        perf = metrics.classification_report(
            df[labeler].apply(lambda x: ' > '.join(x[:level])), 
            df[model].apply(lambda x: ' > '.join(x[:level])), 
            output_dict=True, 
            zero_division=0
        )
    elif level == 0:
        perf = metrics.classification_report(
            df[labeler].apply(lambda x: ' > '.join(x[:])), 
            df[model].apply(lambda x: ' > '.join(x[:])), 
            output_dict=True, 
            zero_division=0
        )
    elif level < 0:
        # for model predicted path: get ancestor at level -1, -2,... but cap at root
        df['tmp_model'] = df[model].apply(lambda x: x[:level] if len(x[:level]) > 0 else x[:1])
        # for ground truth: also get ancestor at level -1, -2,... but cap at root
        df['tmp_label'] = df[labeler].apply(lambda x: x[:level] if len(x[:level]) > 0 else x[:1])
        
        perf = metrics.classification_report(
            df['tmp_label'].apply(lambda x: ' > '.join(x)),
            df['tmp_model'].apply(lambda x: ' > '.join(x)),
            output_dict=True, 
            zero_division=0
        )
        # remove tmp
        del df['tmp_model']
        del df['tmp_label']
    perf = pd.DataFrame(perf).T
    perf = perf.reset_index().rename(columns={'index':'id'})
    perf['level'] = level
    return perf 

def get_df_outfile(dataset):
    if dataset == 'mturk':
        df_data = pd.read_json(
            dvc.api.get_url(
                'data/wish_products/wish-mturk-labelled-09202022-clean-joinedlance.json',
                repo='git@github.com:ContextLogic/multitask-llm-rnd.git'
            ), lines=True
        )
        output_file = os.path.join(os.path.dirname(__file__), 'eval_mturk.xlsx')
    elif dataset == 'offshore':
        df_data = pd.read_json(
            dvc.api.get_url(
                'data/wish_products_internallabel/wish_products_offshore_labelled_processed.json',
                repo='git@github.com:ContextLogic/multitask-llm-rnd.git'
            ), lines=True
        )
        output_file = os.path.join(os.path.dirname(__file__), 'eval_offshore.xlsx')
    elif dataset == 'offshore-validated':
        df_data = pd.read_json(
            dvc.api.get_url(
                'data/wish_products_internallabel/wish_products_offshore_labelled_validated_processed.json',
                repo='git@github.com:ContextLogic/multitask-llm-rnd.git'
            ), lines=True
        )
        output_file = os.path.join(os.path.dirname(__file__), 'eval_offshore_validated.xlsx')
    else:
        raise NotImplemented()
    return df_data, output_file

def save_eval_results(perm, output_file, df_pred, output_file_aggregate, version, dataset):
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', 
                if_sheet_exists='replace') as writer:
            perm.to_excel(writer, index=False, sheet_name=f'{version}_metrics')
            df_pred.to_excel(writer, index=False, sheet_name=f'{version}_preds')
    except:
        with pd.ExcelWriter(output_file, engine='openpyxl', mode='w') as writer:
            perm.to_excel(writer, index=False, sheet_name=f'{version}_metrics')
            df_pred.to_excel(writer, index=False, sheet_name=f'{version}_preds')
    
    tmp = perm[perm['id'].apply(lambda x: x in ['accuracy', 'macro avg', 'weighted avg'])].copy()
    tmp['dataset'] = dataset
    if os.path.exists(output_file_aggregate) and len(pd.read_csv(output_file_aggregate)) > 0:
        tmp.to_csv(output_file_aggregate, mode='a', index=False, header=False)
    else:
        tmp.to_csv(output_file_aggregate, mode='a', index=False, header=True)

def read_json_helper(infer_path):
    try:
        df_pred = pd.read_json(infer_path, lines=True)
    except:
        recs = []
        with open(infer_path, 'r') as f:
            for l in f:
                if len(l) > 2:
                    try:
                        recs.append(json.loads(l))
                    except:
                        recs.append(json.loads(l.split('}{')[0] + '}'))
                        recs.append(json.loads('{' + l.split('}{')[1]))
        df_pred = pd.DataFrame(recs)
    
    df_pred = df_pred.sort_values('batch_indices')
    return df_pred

def main(args):
    """
    python notebooks/eval/eval_from_file.py --dataset offshore-validated \
        --input models/product_title_multitask_multimodal/version_1/clm-epoch=0-step=75000--wish_offshore_validated_wclip--multimodal.json \
        --version mm_clm_clip_v0.1

    Save metrics to notebooks/eval/*.xlsx
    """
    # %%
    df_data, output_file = get_df_outfile(args.dataset)
    output_file_aggregate = os.path.join(os.path.dirname(__file__), 'eval_aggregate.csv')
    infer_path = args.input
    version = args.version

    if version == 'lance':
        perm = pd.concat([perf_eval_util(df_data, level=i, col='lance_predicted_category') for i in [1, 2, 0, -1, -2]])
        df_pred = df_data[['title', 'category', 'lance_predicted_category']]
    else:
        df_pred = read_json_helper(infer_path)
        # %%
        df_pred = df_pred[df_pred.rank_indices == 0].sort_values('batch_indices')

        # %%
        assert len(df_data) == len(df_pred)

        # %%
        df_data[f'{version}_predicted_category'] = df_pred['prediction_decoded'].tolist()

        # %%
        df_data[f'{version}_predicted_category'] = df_data[f'{version}_predicted_category'].apply(lambda x: x.split(' > '))

        # %%
        perm = pd.concat([perf_eval_util(df_data, level=i, col=f'{version}_predicted_category') for i in [1, 2, 0, -1, -2]])
        df_pred = df_data[['title', 'category', f'{version}_predicted_category']]
    
    perm['model_version'] = args.version
    save_eval_results(perm, output_file, df_pred, output_file_aggregate, version, args.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='mturk|offshore|offshore-validated')
    parser.add_argument('--input', help='input inference file used')
    parser.add_argument('--version', help='model version used, "lance" being lance\'s model')

    args = parser.parse_args()
    assert args.dataset in ['mturk', 'offshore', 'offshore-validated']
    main(args)