#%%
import pandas as pd
import numpy as np
import dvc.api
import os
import yaml
from sklearn.metrics import classification_report
from pprint import pprint

## inference script
# python main_inference_multimodal.py \
#         --model=LLM_Inference_Multimodal \
#         --model.llm_type="seqclf" \
#         --model.ckpt_path="models/multitask_multimodal_multilingual/version_11/epoch=3-step=776.ckpt" \
#         --model.config_path="models/multitask_multimodal_multilingual/version_11/config.yaml" \
#         --model.task="seqclf_singlemodal_wishquery2tax" \
#         --model.output_dir="models/multitask_multimodal_multilingual/version_11" \
#         --model.write_interval="batch" \
#         --data=JSONListData \
#         --data.llm_type="seqclf" \
#         --data.label_map_file="datasets/taxonomy/wish_v1.2.1_newtax_allpaths_withunknown.txt" \
#         --data.label_type="multilabel_taxonomy" \
#         --data.data_source_yaml_path="datasets/multimodal_multitask/wish_labelled_query_offshore_test_V2.yaml" \
#         --data.input_dict="{'template': '{query}', 'task_prefix': 'Classify query: '}" \
#         --data.output_dict="{'template': '{query}'}" \
#         --data.data_source_type="dvc" \
#         --data.model_name="microsoft/Multilingual-MiniLM-L12-H384" \
#         --data.batch_size=50 \
#         --data.max_length=50 \
#         --data.num_workers=0 \
#         --data.overwrite_cache=true \
#         --data.force_download_hfdata=true \
#         --trainer.logger=false \
#         --trainer.enable_checkpointing=false \
#         --trainer.accelerator gpu \
#         --trainer.strategy ddp

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# configurable
data_config_relpath = "datasets/multimodal_multitask/wish_labelled_query_offshore_test_V2.yaml"
data_split = "test"
data_source_type = "dvc"
inference_output_relpath = "models/multitask_multimodal_multilingual/version_11/seqclf-epoch=3-step=776--wish_labelled_query_offshore_test_V2--test.json"

tax_data_config = {
    "path": "datasets/data/taxonomy/wish_newtax.json",
    "repo": "git@github.com:ContextLogic/multitask-llm-rnd.git",
    "rev": None
}
baseline_data_config = {
    "path": "datasets/data/query/wish_queries_inferred_newtax.json",
    "repo": "git@github.com:ContextLogic/multitask-llm-rnd.git",
    "rev": None
}

for use_lang in ['en', 'all']:
    for min_prob, min_baseline_weight in [(-1, -1)]:
        for eval_top_k in [3]:
            data_config = yaml.safe_load(open(os.path.join(main_path, data_config_relpath), 'r'))
            assert data_split in data_config and len(data_config[data_split]) == 1
            if data_source_type == "dvc":
                df_in = pd.read_json(dvc.api.get_url(**data_config[data_split][0]), lines=True)
            else:
                raise NotImplemented()
            df_out = pd.read_json(os.path.join(main_path, inference_output_relpath), lines=True)
            assert len(df_out) == len(df_in)
            df_baseline = pd.read_json(dvc.api.get_url(**baseline_data_config), lines=True)
            df_tax = pd.read_json(dvc.api.get_url(**tax_data_config), lines=True)

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
            df_out_group = df_out.sort_values('batch_indices')
            df_out_group['prediction_decoded'] = df_out_group['preds'].apply(lambda x: [i[0] for i in x[:eval_top_k]])
            df_out_group['prob'] = df_out_group['preds'].apply(lambda x: [i[1] for i in x[:eval_top_k]])
            # df_out_group = df_out.groupby('batch_indices').agg(
            #     {
            #         'prediction_decoded': lambda x: [i for i in x], 
            #         'prob': lambda x: [i for i in x]
            #     }).sort_index()

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
                    assert j in valid_paths or j == 'unknown'

            valid_paths_rev = {valid_paths[i]: i for i in valid_paths}
            valid_paths_list = [valid_paths_rev[i] for i in range(len(valid_paths_rev))]

            df_join = df_in.copy()[['query', 'lang', 'sample_method', 'gmv', 'cnt', 
                    'query_classification_lists']].rename(
                columns={'query_classification_lists': 'y_true'})
            df_join['y_pred'] = df_out_group['prediction_decoded'].tolist()
            df_join['y_prob'] = df_out_group['prob'].tolist()

            df_join_baseline = df_join.merge(df_baseline[['query', 'y_baseline', 'y_baseline_prob']], 
                on='query', how='inner')


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
                            if j == 'unknown':
                                # ignore any predictions less confident than unknown
                                break
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


            df_perf, df_perf_agg = calculate_metrics(df_join, min_prob, valid_paths, y_true_name="y_true", 
                    y_pred_name="y_pred", y_prob_name="y_prob")
            df_sub_perf, df_sub_perf_agg = calculate_metrics(df_join_baseline, min_prob, valid_paths, 
                    y_true_name="y_true", y_pred_name="y_pred", y_prob_name="y_prob")
            df_sub_perf_baseline, df_sub_perf_agg_baseline = calculate_metrics(df_join_baseline, min_baseline_weight, 
                    valid_paths, y_true_name="y_true", y_pred_name="y_baseline", y_prob_name="y_baseline_prob")

            if use_lang == 'en':
                df_join_baseline_en = df_join_baseline[df_join_baseline.lang == 'en']
            else:
                df_join_baseline_en = df_join_baseline
            df_sub_perf_en, df_sub_perf_agg_en = calculate_metrics(df_join_baseline_en, min_prob, valid_paths, 
                    y_true_name="y_true", y_pred_name="y_pred", y_prob_name="y_prob")
            df_sub_perf_baseline_en, df_sub_perf_agg_baseline_en = calculate_metrics(df_join_baseline_en, min_baseline_weight, 
                    valid_paths, y_true_name="y_true", y_pred_name="y_baseline", y_prob_name="y_baseline_prob")

            recs = []
            for i in df_join_baseline_en.to_dict('records'):
                i['y_baseline_jc'] = len(set(i['y_true'][:eval_top_k]).intersection(set(i['y_baseline'][:eval_top_k]))) / \
                    len(set(i['y_true'][:eval_top_k]).union(set(i['y_baseline'][:eval_top_k])))
                i['y_pred_jc'] = len(set(i['y_true']).intersection(set(i['y_pred'][:eval_top_k]))) / \
                    len(set(i['y_true'][:eval_top_k]).union(set(i['y_pred'][:eval_top_k])))
                recs.append(i)
            df_join_baseline_en = pd.DataFrame(recs)

            # writer = pd.ExcelWriter(outfile_path)

            # # Write each dataframe to a different worksheet.
            # df_sub_perf_agg_en.to_excel(writer, sheet_name='V1model_metrics', index=False)
            # df_sub_perf_agg_baseline_en.to_excel(writer, sheet_name='V0baseline_metrics', index=False)
            # df_join_baseline_en.to_excel(writer, sheet_name='Results', index=False)

            # # Close the Pandas Excel writer and output the Excel file.
            # writer.close()


            print(f"\nuse_lang {use_lang} min_prob {min_prob}, min_baseline_weight {min_baseline_weight}, eval_top_k {eval_top_k}")
            print("\nbaseline\n")
            print(df_sub_perf_agg_baseline_en.to_markdown(index=False))
            print("\nV1\n")
            print(df_sub_perf_agg_en.to_markdown(index=False))

#%%