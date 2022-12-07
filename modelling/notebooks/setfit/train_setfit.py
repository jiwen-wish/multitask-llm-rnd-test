# %%
import sys 
sys.path.append('../..')

# %%
from main_utils import LLM_EmbedData

# %%
from datasets import load_dataset, Dataset
from sentence_transformers.losses import CosineSimilarityLoss

from setfit import SetFitModel, SetFitTrainer

# %%
data_demo = load_dataset("SetFit/SentEval-CR")

# %%
data_demo['train'][:2]

# %%
data = LLM_EmbedData(
    '../../datasets/product_title_embedding/wish-tahoe-openai-reversegen.yaml', 
    model_name='sentence-transformers/sentence-t5-base'
)

data_val = LLM_EmbedData(
    '../../datasets/product_title_embedding/wish-offshore-test.yaml', 
    model_name='sentence-transformers/sentence-t5-base'
)

# %%
data.prepare_data()
data_val.prepare_data()

# %%
ds = data.get_hf_dataset()
ds_val = data_val.get_hf_dataset()

# %%
from sklearn.preprocessing import LabelEncoder

# %%
le = LabelEncoder()

# %%
le.fit(sorted(list(
    set(ds['train'][:]['text_output'] + ds['val'][:]['text_output'] + \
        ds['test'][:]['text_output'] + ds_val['test'][:]['text_output'])
)))

# %%
def transform(examples):
    outs = {}
    outs['text'] = [i.split('Embed product: ')[1] for i in examples['text_input']]
    outs['label'] = le.transform([i for i in examples['text_output']]).tolist()
    outs['label_text'] = [i.split('Embed taxonomy: ')[1] for i in examples['text_output']]
    return outs

# %%
ds.set_transform(transform)
ds_val.set_transform(transform)

# %%
import pandas as pd

# %%
train_ds = Dataset.from_pandas(pd.DataFrame(ds['train'][:]))
val_ds = Dataset.from_pandas(pd.DataFrame(ds_val['test'][:]))

print(len(train_ds), len(val_ds))

# %%
model = SetFitModel.from_pretrained("sentence-transformers/sentence-t5-base")

# Create trainer
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    loss_class=CosineSimilarityLoss,
    batch_size=16,
    num_iterations=5, # Number of text pairs to generate for contrastive learning
    num_epochs=1 # Number of epochs to use for contrastive learning
)

# %%
trainer.train()
#%%
metrics = trainer.evaluate()
print(metrics)
# %%



