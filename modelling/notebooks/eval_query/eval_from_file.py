#%%
import pandas as pd
import numpy as np
import dvc.api
import os
import yaml
from sklearn.metrics import classification_report
from pprint import pprint


main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
outfile_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'eval_offshore.xlsx'))

# configurable
data_config_relpath = "datasets/multimodal_multitask/wish_labelled_query_offshore_test.yaml"
data_split = "test"
data_source_type = "dvc"
inference_output_relpath = "models/product_title_multitask_multimodal/version_2/clm-epoch=2-step=80080--wish_labelled_query_offshore_test--test.json"
num_return_sequences = 3
tax_data_config = {
    "path": "data/taxonomy/wish_newtax.json",
    "repo": "git@github.com:junwang-wish/query_understanding_data.git",
    "rev": None
}
baseline_data_config = {
    "path": "data/query/wish_queries_inferred_newtax.json",
    "repo": "git@github.com:junwang-wish/query_understanding_data.git",
    "rev": None
}
# don't remove categories if prob/weight too small
min_prob = -1
min_baseline_weight = -1
# don't truncate categories if predict more than 3
eval_top_k = 10000 

if __name__ == "__main__":
    pass

    #%%
    data_config = yaml.safe_load(open(os.path.join(main_path, data_config_relpath), 'r'))
    assert data_split in data_config and len(data_config[data_split]) == 1
    if data_source_type == "dvc":
        df_in = pd.read_json(dvc.api.get_url(**data_config[data_split][0]), lines=True)
    else:
        raise NotImplemented()
    df_out = pd.read_json(os.path.join(main_path, inference_output_relpath), lines=True)
    assert len(df_out) == int(num_return_sequences * len(df_in))
    df_baseline = pd.read_json(dvc.api.get_url(**baseline_data_config), lines=True)
    df_tax = pd.read_json(dvc.api.get_url(**tax_data_config), lines=True)

    # %%
    df_in['query_classification_lists'] = df_in['query_classification_lists'].apply(
        lambda x: [i.lower() for i in x]
    )
    df_tax['category_path'] = df_tax['category_path'].str.lower()
    taxstrid2path = {}
    for i in df_tax.to_dict('records'):
        if len(i['category_path']) > 0:
            taxstrid2path[str(i['id'])] = i['category_path']
    df_baseline['y_baseline'] = df_baseline['categories'].apply(
        lambda x: [taxstrid2path[i] for i in x.split(',')]
    )
    df_baseline['y_baseline_prob'] = df_baseline['weights'].apply(
        lambda x: [float(i) for i in x.split(',')]
    )
    df_out_group = df_out.groupby('batch_indices').agg(
        {
            'prediction_decoded': lambda x: [i for i in x], 
            'prob': lambda x: [i for i in x]
        }).sort_index()
    #%%
    valid_paths = {}
    valid_paths_rev = {}
    for i in df_tax.to_dict('records'):
        if len(i['category_path']) > 0:
            valid_paths[i['category_path']] = len(valid_paths)
    for i in df_in.to_dict('records'):
        for j in i['query_classification_lists']:
            assert j in valid_paths
    for i in df_out_group.to_dict('records'):
        for j in i['prediction_decoded']:
            assert j in valid_paths

    valid_paths_rev = {valid_paths[i]: i for i in valid_paths}
    valid_paths_list = [valid_paths_rev[i] for i in range(len(valid_paths_rev))]

    df_join = df_in.copy()[['query', 'lang', 'sample_method', 'gmv', 'cnt', 
            'query_classification_lists']].rename(
        columns={'query_classification_lists': 'y_true'})
    df_join['y_pred'] = df_out_group['prediction_decoded'].tolist()
    df_join['y_prob'] = df_out_group['prob'].tolist()

    df_join_baseline = df_join.merge(df_baseline[['query', 'y_baseline', 'y_baseline_prob']], 
        on='query', how='inner')

    # %%
    def calculate_metrics(df_join, min_prob, valid_paths, y_true_name="y_true", 
            y_pred_name="y_pred", y_prob_name="y_prob"):
        perfs = []
        for depth_constraint in [1, 2, 3, 4, 5]:
            # build label indicator array
            y_true_indicator = np.zeros((len(df_join), len(valid_paths)))
            y_pred_indicator = np.zeros((len(df_join), len(valid_paths)))
            for ind, i in enumerate(df_join.to_dict('records')):
                for j in i[y_true_name][:eval_top_k ]: 
                    path = " > ".join(j.split(" > ")[:depth_constraint])
                    y_true_indicator[ind, valid_paths[path]] = 1.
                for j, j_prob in list(zip(i[y_pred_name], i[y_prob_name]))[:eval_top_k ]:
                    path = " > ".join(j.split(" > ")[:depth_constraint])
                    if j_prob >= min_prob:
                        y_pred_indicator[ind, valid_paths[path]] = 1.
            perf = pd.DataFrame(classification_report(
                y_true=y_true_indicator,
                y_pred=y_pred_indicator,
                output_dict=True, 
                zero_division=0,
                target_names=valid_paths_list
            )).T.reset_index().rename(columns={'index':'id'})
            perf['depth_constraint'] = depth_constraint
            perfs.append(perf)
        df_perf = pd.concat(perfs)
        df_perf_agg = df_perf[df_perf['id'] == 'weighted avg']
        return df_perf, df_perf_agg

    # %%
    df_perf, df_perf_agg = calculate_metrics(df_join, min_prob, valid_paths, y_true_name="y_true", 
            y_pred_name="y_pred", y_prob_name="y_prob")
    df_sub_perf, df_sub_perf_agg = calculate_metrics(df_join_baseline, min_prob, valid_paths, 
            y_true_name="y_true", y_pred_name="y_pred", y_prob_name="y_prob")
    df_sub_perf_baseline, df_sub_perf_agg_baseline = calculate_metrics(df_join_baseline, min_baseline_weight, 
            valid_paths, y_true_name="y_true", y_pred_name="y_baseline", y_prob_name="y_baseline_prob")

    #%%
    df_join_baseline_en = df_join_baseline[df_join_baseline.lang == 'en']
    df_sub_perf_en, df_sub_perf_agg_en = calculate_metrics(df_join_baseline_en, min_prob, valid_paths, 
            y_true_name="y_true", y_pred_name="y_pred", y_prob_name="y_prob")
    df_sub_perf_baseline_en, df_sub_perf_agg_baseline_en = calculate_metrics(df_join_baseline_en, min_baseline_weight, 
            valid_paths, y_true_name="y_true", y_pred_name="y_baseline", y_prob_name="y_baseline_prob")
    # %%
    recs = []
    for i in df_join_baseline_en.to_dict('records'):
        i['y_baseline_jc'] = len(set(i['y_true'][:3]).intersection(set(i['y_baseline'][:3]))) / \
            len(set(i['y_true'][:3]).union(set(i['y_baseline'][:3])))
        i['y_pred_jc'] = len(set(i['y_true']).intersection(set(i['y_pred'][:3]))) / \
            len(set(i['y_true'][:3]).union(set(i['y_pred'][:3])))
        recs.append(i)
    df_join_baseline_en = pd.DataFrame(recs)
    #%%
    writer = pd.ExcelWriter(outfile_path)

    # Write each dataframe to a different worksheet.
    df_sub_perf_agg_en.to_excel(writer, sheet_name='V1model_metrics', index=False)
    df_sub_perf_agg_baseline_en.to_excel(writer, sheet_name='V0baseline_metrics', index=False)
    df_join_baseline_en.to_excel(writer, sheet_name='Results', index=False)

    # Close the Pandas Excel writer and output the Excel file.
    writer.close()

    #%%