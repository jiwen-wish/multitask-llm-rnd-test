#%%
import json
import pandas as pd
import dvc.api
import os 

os.system('rm 250_l2stratified_sample/*')

allowed_sessions = ["hao", "hao2", "hao3"]
allowed_sessions = [i.strip().lower() for i in allowed_sessions]

df = pd.read_csv(dvc.api.get_url(
    'datasets/data/query_attr_extract_label/processed/l2stratified_sample_250_query.csv',
    repo='git@github.com:ContextLogic/multitask-llm-rnd.git'
))

df_tax = pd.read_json(dvc.api.get_url(
    'datasets/data/taxonomy/wish_newtax.json',
    repo='git@github.com:ContextLogic/multitask-llm-rnd.git'
), lines=True)

pathtoid = {}
for i in df_tax.to_dict('records'):
    pathtoid[i['category_path']] = int(i['id'])

df_ontology = pd.read_csv(dvc.api.get_url(
    'datasets/data/wish_attr_extract_label/[ontology] wish_top25L2_attributes - 20221219.csv',
    repo='git@github.com:ContextLogic/multitask-llm-rnd.git'
))

gmv_ranked_l2s = """Home Improvement > Lights & Lighting
Home & Garden > Home Decor
Home & Garden > Arts, Crafts & Sewing
Women's Clothing > Tops
Home & Garden > Home Textile
Home & Garden > Kitchen,Dining & Bar
Women's Clothing > Underwear & Sleepwear
Men's Clothing > Tops & Tees
Automobiles & Motorcycles > Interior Accessories
Home & Garden > Pet Products
Women's Clothing > Dresses
Shoes > Women's Shoes
Cellphones & Telecommunications > Mobile Phone Accessories
Home & Garden > Festive & Party Supplies
Cellphones & Telecommunications > Phone Bags & Cases
Tools > Hand Tools
Jewelry & Accessories > Fine Jewelry
Automobiles & Motorcycles > Auto Replacement Parts
Sports > Sneakers
Home & Garden > Garden Supplies
Sports > Fishing
Jewelry & Accessories > Necklaces & Pendants
Watches > Men's Watches
Automobiles & Motorcycles > Motorcycle Accessories & Parts
Jewelry & Accessories > Rings""".split('\n')
gmv_ranked_l2s = [i.strip() for i in gmv_ranked_l2s]

fashion_gmv_ranked_l2s = [
    "Women's Clothing > Tops",
    "Women's Clothing > Underwear & Sleepwear",
    "Men's Clothing > Tops & Tees",
    "Women's Clothing > Dresses"
]

#%%
start_port = 8080
proj_folder = '250_l2stratified_sample'
with open('start_prodigy.sh', 'w') as f_shell:
    for l2 in set(df.top_valid_l2):
        if True: # l2 in fashion_gmv_ranked_l2s:
            attributes = sorted(list(set(df_ontology[df_ontology.wish_L2 == l2].attribute_name.tolist())))
            fname = f'{proj_folder}/inputs_taxid_{pathtoid[l2]}_l2_{l2}.jsonl'
            with open(fname, 'w') as f:
                for l in df[df.top_valid_l2 == l2].to_dict('records'):
                    f.write(json.dumps(
                        {
                            "text": l['query'], 
                            "meta": {"l2": l['top_valid_l2']}
                        }
                    ) + '\n')
            f_shell.write(f"""PRODIGY_PORT={start_port} PRODIGY_ALLOWED_SESSIONS={",".join(allowed_sessions)} PRODIGY_LOGGING=basic prodigy ner.manual {proj_folder} blank:en "{fname}" --label "{",".join(attributes)}\"""" + "\n")
            start_port += 1
# %%
