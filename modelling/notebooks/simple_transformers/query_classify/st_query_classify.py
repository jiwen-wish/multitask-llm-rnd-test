# %%
from simpletransformers.classification import (
    MultiLabelClassificationModel, MultiLabelClassificationArgs
)
import pandas as pd
import logging
import numpy as np
import dvc.api
from tqdm import tqdm
import yaml
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', help='Transformer model type', type=str, default='xlmroberta', required=False)
parser.add_argument('--data_config', help='Yaml path for train/val data', type=str, required=True)
parser.add_argument('--load_ckpt', help='Folder path for ckpt', type=str, required=True)
parser.add_argument('--output_dir', help='Folder path for saving ckpt', type=str, required=True)
parser.add_argument('--truncate_depth', help='Truncate taxonomy depth at', type=int, required=True)
parser.add_argument('--n_gpu', help='Number of GPUs', type=int, default=1, required=False)
parser.add_argument('--num_train_epochs', help='Number of epochs', type=int, default=4, required=False)
args = parser.parse_args()
print(args)

# %%
data_config = yaml.safe_load(open(args.data_config, 'r'))
df_train = pd.concat([pd.read_json(dvc.api.get_url(
    i['path'],
    repo=i['repo']
), lines=True) for i in data_config['train']])
df_val = pd.concat([pd.read_json(dvc.api.get_url(
    i['path'],
    repo=i['repo']
), lines=True) for i in data_config['val']])

#%%
df_train['category'] = df_train['category'].apply(lambda x: ' > '.join(x.split(' > ')[:args.truncate_depth]))
df_val['category'] = df_val['category'].apply(lambda x: ' > '.join(x.split(' > ')[:args.truncate_depth]))

# %%
df_tax = pd.read_json(dvc.api.get_url(
    'datasets/data/taxonomy/wish_newtax.json',
    repo='git@github.com:ContextLogic/multitask-llm-rnd.git'
), lines=True)
df_tax = df_tax[(df_tax.category_path.apply(len) > 0) & \
    (df_tax.category_path.apply(lambda x: len(x.split(' > ')) <= args.truncate_depth))]

# %%
LABEL_SET = sorted(df_tax.category_path.str.lower().str.strip().tolist()) + ['unknown']

# %%
LABEL_NAME_TO_ID = {i: ind for ind, i in enumerate(LABEL_SET)}
# print(LABEL_NAME_TO_ID)
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

labs = []
for i in tqdm(df_train_group['category'].tolist()):
    labs.append(categories2labels(i))
df_train_group['labels'] = labs 

labs = []
for i in tqdm(df_val_group['category'].tolist()):
    labs.append(categories2labels(i))
df_val_group['labels'] = labs 

df_train_group['text'] = df_train_group['query']
df_val_group['text'] = df_val_group['query']

# %%
# Optional model configuration
model_args = MultiLabelClassificationArgs(
    no_cache=True,
    num_train_epochs=args.num_train_epochs, 
    use_multiprocessing=False,
    use_multiprocessing_for_evaluation=False,
    save_steps=-1,
    save_model_every_epoch=False,
    save_eval_checkpoints=False,
    n_gpu=args.n_gpu,
    train_batch_size=100, 
    evaluate_during_training=True,
    evaluate_during_training_verbose=True,
    use_cached_eval_features=False,
    output_dir=args.output_dir,
    best_model_dir=f'{args.output_dir}/best_model',
)

def generate_pos_weight(labels):
    n_classes = labels.shape[1]
    pos_weights = np.zeros(n_classes)
    for i in tqdm(range(n_classes)):
        class_labels = labels[:, i]
        unique, counts = np.unique(class_labels, return_counts=True)
        class_counts = dict(zip(unique, counts))
        if 0 not in class_counts:
            class_counts[0] = 1
        if 1 not in class_counts:
            class_counts[1] = 1
        pos_weights[i] = class_counts[0] / class_counts[1]
    pos_weights = np.log(1 + pos_weights)
    return pos_weights

# Create a MultiLabelClassificationModel
pos_weight = generate_pos_weight(np.stack([np.array(l) for l in tqdm(df_train_group['labels'])]))
print('pos_weight stats: ', pos_weight.mean(), pos_weight.min(), pos_weight.max(), np.median(pos_weight))
try:
    model = MultiLabelClassificationModel(
        args.model_type,
        args.load_ckpt,
        num_labels=len(LABEL_NAME_TO_ID),
        pos_weight=pos_weight.tolist(),
        args=model_args,
        ignore_mismatched_sizes=True
    )
except:
    import pdb; pdb.set_trace()

# %%
# Train the model
model.train_model(train_df=df_train_group, eval_df=df_val_group)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(
    df_val_group
)
print(result)
# Make predictions with the model
predictions, raw_outputs = model.predict(["dildo", "cake", "iphone"])
print(raw_outputs.argmax(1))
# %%



