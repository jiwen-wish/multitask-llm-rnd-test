# %%
from simpletransformers.classification import (
    MultiLabelClassificationModel, MultiLabelClassificationArgs
)
import pandas as pd
import logging
import numpy as np


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# %%
import dvc.api

# %%
# df_train = pd.concat([
#     pd.read_json(dvc.api.get_url(
#         'datasets/data/query_label/processed2/Human_Labelled_Query_Classification_Train.json',
#         repo='git@github.com:ContextLogic/multitask-llm-rnd.git'
#     ), lines=True), 
# ])
df_train = pd.concat([
    pd.read_json(dvc.api.get_url(
        'datasets/data/query_label/processed3/Mixed_Human_Inferred_Query_Classification_Train_DedupOverlap.json',
        repo='git@github.com:ContextLogic/multitask-llm-rnd.git'
    ), lines=True), 
    pd.read_json(dvc.api.get_url(
        'datasets/data/query_label/processed3/OnlyHuman_Labelled_Query_Classification_Train_DedupOverlap.json',
        repo='git@github.com:ContextLogic/multitask-llm-rnd.git'
    ), lines=True), 
    pd.read_json(dvc.api.get_url(
        'datasets/data/query_label/processed3/OnlyInferred_Query_Classification_Train_DedupOverlap.json',
        repo='git@github.com:ContextLogic/multitask-llm-rnd.git'
    ), lines=True), 
])
# df_val = pd.read_json(dvc.api.get_url(
#     'datasets/data/query_label/processed2/Human_Labelled_Query_Classification_Val.json',
#     repo='git@github.com:ContextLogic/multitask-llm-rnd.git'
# ), lines=True)
df_val = pd.read_json(dvc.api.get_url(
    'datasets/data/query_label/processed3/Human_Labelled_Query_Classification_Val_DedupOverlap.json',
    repo='git@github.com:ContextLogic/multitask-llm-rnd.git'
), lines=True)
df_test = pd.read_json(dvc.api.get_url(
    'datasets/data/query_label/processed/Offshore_Labelled_Query_Classification_Test_V2.json',
    repo='git@github.com:ContextLogic/multitask-llm-rnd.git'
), lines=True)

#%%
df_train['category'] = df_train['category'].apply(lambda x: x.split(' > ')[0])
df_val['category'] = df_val['category'].apply(lambda x: x.split(' > ')[0])

# %%
df_tax = pd.read_json(dvc.api.get_url(
    'datasets/data/taxonomy/wish_newtax.json',
    repo='git@github.com:ContextLogic/multitask-llm-rnd.git'
), lines=True)
df_tax = df_tax[(df_tax.category_path.apply(len) > 0) & (df_tax.category_path.apply(lambda x: len(x.split(' > ')) == 1))]

# %%
LABEL_SET = sorted(df_tax.category_path.str.lower().str.strip().tolist()) + ['unknown']

# %%
LABEL_NAME_TO_ID = {i: ind for ind, i in enumerate(LABEL_SET)}
print(LABEL_NAME_TO_ID)
# %%
df_train_group = df_train.groupby('query').agg({'category': lambda x: [i for i in x]}).reset_index()
df_val_group = df_val.groupby('query').agg({'category': lambda x: [i for i in x]}).reset_index()

# %%
def categories2labels(cats):
    if len(cats) == 0:
        cats = ['unknown']
    labs = [0] * len(LABEL_NAME_TO_ID)
    for c in cats:
        labs[LABEL_NAME_TO_ID[c]] = 1
    return labs

# %%
df_train_group = df_train_group[df_train_group.category.apply(lambda x: '' not in x)]
df_val_group = df_val_group[df_val_group.category.apply(lambda x: '' not in x)]
df_train_group['labels'] = df_train_group['category'].apply(categories2labels)
df_val_group['labels'] = df_val_group['category'].apply(categories2labels)
df_train_group['text'] = df_train_group['query']
df_val_group['text'] = df_val_group['query']

# %%
# Optional model configuration
model_args = MultiLabelClassificationArgs(
    num_train_epochs=4, 
    use_multiprocessing=False,
    use_multiprocessing_for_evaluation=False,
    n_gpu=1,
    train_batch_size=100, 
    evaluate_during_training=True,
    evaluate_during_training_verbose=True,
    use_cached_eval_features=False
)

def generate_pos_weight(labels):
    n_classes = labels.shape[1]
    pos_weights = np.zeros(n_classes)
    for i in range(n_classes):
        class_labels = labels[:, i]
        unique, counts = np.unique(class_labels, return_counts=True)
        class_counts = dict(zip(unique, counts))
        if 0 not in class_counts:
            class_counts[0] = 1
        if 1 not in class_counts:
            class_counts[1] = 1
        pos_weights[i] = class_counts[0] / class_counts[1]
    return np.log(1 + pos_weights)

# Create a MultiLabelClassificationModel
pos_weight = generate_pos_weight(np.array(df_train_group['labels'].tolist()))
print(pos_weight.tolist())
model = MultiLabelClassificationModel(
    "xlmroberta",
    "xlm-roberta-base",
    num_labels=len(LABEL_NAME_TO_ID),
    pos_weight=pos_weight.tolist(),
    args=model_args,
)


# %%
# !rm -rf cache_dir  outputs  runs 

# %%
# Train the model
model.train_model(train_df=df_train_group, eval_df=df_val_group)

# %%

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(
    df_val_group
)
print(result)
# Make predictions with the model
predictions, raw_outputs = model.predict(["dildo", "cake", "iphone"])
print(raw_outputs.argmax(1))
# %%



