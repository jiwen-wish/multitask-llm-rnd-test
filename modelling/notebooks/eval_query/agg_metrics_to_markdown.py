#%% 
import pandas as pd 
import os 

input_excel = os.path.join(os.path.dirname(__file__), "eval_offshore.xlsx")
output_markdown = os.path.join(os.path.dirname(__file__), "README.md")

df = pd.read_excel(input_excel, sheet_name=None)

#%%
df_out = []
for i in df:
    if '_metrics' in i:
        tmp = df[i]
        tmp['model'] = i.replace('_metrics', '')
        df_out.append(tmp)
df_out = pd.concat(df_out)
str_out = df_out.to_markdown(index=False)
# %%
with open(output_markdown, 'w') as f:
    f.write(str_out)
#%%