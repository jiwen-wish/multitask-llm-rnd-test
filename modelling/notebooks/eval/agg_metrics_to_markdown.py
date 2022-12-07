#%% 
import pandas as pd 
import os 

input_csv = os.path.join(os.path.dirname(__file__), "eval_aggregate.csv")
output_markdown = os.path.join(os.path.dirname(__file__), "README.md")

df = pd.read_csv(input_csv)

#%%
df_out = df[(df['id'] == 'weighted avg') & (df['dataset'] == 'offshore-validated')].sort_values(
    ['dataset', 'level', 'model_version'])
str_out = df_out.to_markdown(index=False)
# %%
with open(output_markdown, 'w') as f:
    f.write(str_out)