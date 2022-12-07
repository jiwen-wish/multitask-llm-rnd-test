# multitask-llm-rnd modelling
RnD modelling repo for query understanding and more generally content understanding tasks using large language models (llm).

## High-level approach [Slide](https://docs.google.com/presentation/d/1W1jvTWK43Pxhx22XDuF7BuNuifnOEI_eLtQUeqDZraw/edit?usp=sharing), [Brainstorm](https://docs.google.com/document/d/1fsskXMGYeHsPGHaJYBrjus9Y7pDFdfG9cys2ehdqxQY/edit?usp=sharing)
- Simple approach based on language modelling
- Exploit 0-shot generalization to unseen domain due to pretraining
- Extensible and flexible to utilize newly available metadata (taxonomy) and modality (image, latent features)

## Project: Query Classification v1
- Given a [raw user query](https://docs.google.com/spreadsheets/d/1rUQcUzla8rCsKfmzVt4smUJ8x0d3fasKAa5XiPEqWkQ/edit?usp=sharing), predict top-K most likely categories based on [v1.2.1 Taxonomy](https://docs.google.com/spreadsheets/d/17OXUosk0RZzSSQFyBkwdP-U9WFDPgX8Wxxn6r22UuvM/edit?usp=sharing), using [v2 guideline](https://docs.google.com/document/d/1RC9LcTOuOGGHPMMOfVemHo1__0uAyobSgBeiFOTiZGg/edit?usp=sharing) labelled by internal offshore team
- A query can be classified as multiple internal and lead nodes in the taxonomy tree
- Allow for future extensions to incorporate other information and modalities, but focused on text only for now

## Important Notes
- Read [dvc doc](https://dvc.org/doc) before running any `dvc` commands in this repo or [query_understand_data](https://github.com/junwang-wish/query_understanding_data) repo. Otherwise, incorrect commands might wipe all trained models and data and are irreversible (not what we want).

## Reproduce
This repo should be run on [Dgx1](https://wiki.wish.site/display/~ldeng/DGX1) inside a [VScode Dev Docker Container](https://code.visualstudio.com/docs/remote/containers) to ensure full reproducibility.

### 1. Have correct .bashrc and .ssh under $HOME directory

On your EC2 instance, have credentials needed to access AWS S3, Tahoe, Gitlab in ```$HOME/.bashrc```, as well as Cache directories to use in ```$HOME/.bashrc```, and Github access keys in ```$HOME/.ssh``` folder, since they will be mounted to the dev container to pass authentification variables:

```shell
junwang@junwang-ec2:~/query_understanding_data$ ls $HOME/.ssh

authorized_keys  id_ecdsa  id_ecdsa.pub  id_rsa  id_rsa.pub  known_hosts


junwang@junwang-ec2:~/query_understanding_data$ tail -n 40 $HOME/.bashrc

# add AWS
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_KEY=...
export AWS_SECRET=...

# add Tahoe
export TAHOE_USERNAME=junwang
export TAHOE_PASSWORD=...

# add Gitlab
export GITLAB_USERNAME=junwang-wish
export GITLAB_ACCESS_TOKEN=...

# add Huggingface cache dir
export HF_DATASETS_CACHE="/data/junwang/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/data/junwang/.cache/huggingface/transformers"

# add general cache dir
export GENERAL_CACHE="/data/junwang/.cache/general"
```

### 2. Reopen repo in Container using VScode
Dependencies needed to run data pipelines are installed during docker building.
- ```Shift-Command-P``` on Mac or ```Ctrl-Shift-P``` on Windows to open VScode menu
- Search and click ```Remote-Containers: Reopen in Container```
- Wait for dev docker build if it is first-time usage, otherwise existing built docker will be reused

### 3. `dvc pull` checkpoints and load them
Read [dvc doc](https://dvc.org/doc) before running any `dvc` commands in this repo or [query_understand_data](https://github.com/junwang-wish/query_understanding_data) repo. Otherwise, incorrect commands might wipe all trained models and data and are irreversible (not what we want).

To download a specific checkpoint, i.e. `models/product_title_taxonomy_sequence_classification/version_0/epoch=4-step=59300.ckpt` which is tracked by [models/product_title_taxonomy_sequence_classification/version_0/epoch=4-step=59300.ckpt.dvc](models/product_title_taxonomy_sequence_classification/version_0/epoch=4-step=59300.ckpt.dvc), run
- `dvc pull models/product_title_taxonomy_sequence_classification/version_0/epoch=4-step=59300.ckpt`

These models are not saved to github due to limit, but instead saved to `dvc` which links to s3, specifically `s3://structured-data-dev/junwang_query_understand_model/dvc` which is specified in [.dvc/config](.dvc/config)

To load checkpoint, refer to [notebooks/model/load_checkpoint.ipynb](notebooks/model/load_checkpoint.ipynb)

### 4. Training [models](models)
`README.md` under subfolders of [models](models) contain one-line training commands to reproduce existing model's training. To train new models, create new subfolders `version_*` under correct subdirectory and run similar one-line training commands.

### 5. Inference models
Understand how [run_inference.sh](run_inference.sh) works and run the full script or subset of it.

### 6. Evaluate models
After inference, go to [notebooks/eval](notebooks/eval) and use the following scripts (they contain sample command usage in code as comment)
- [notebooks/eval/eval_from_file.py](notebooks/eval/eval_from_file.py) for clm (conditional language modelling) models.
- [notebooks/eval/eval_from_file_emb.py](notebooks/eval/eval_from_file_emb.py) for emb (embedding) models.
- [notebooks/eval/eval_from_file_seqclf.py](notebooks/eval/eval_from_file_seqclf.py) for seqclf (sequence classification) models.

### 7. Develop new models
Modify the following scripts in non-breaking way (if add new tricks, make sure the default old behavior don't change)
- [main_conditional_lm.py](main_conditional_lm.py) for clm (conditional language modelling) models.
- [main_denoise_lm.py](main_denoise_lm.py) for dlm (denoising language modelling) models.
- [main_embedding.py](main_embedding.py) for emb (embedding) models.
- [main_seqclassify.py](main_seqclassify.py) for seqclf (sequence classification) models.
- [main_multitask.py](main_multitask.py) for multitask models.

Or create new `main_*.py` files.

### 8. Use new dataset
Understand formats of `*.yaml` (data yamls) under subdirectories of `datasets`. They import data created from [query_understand_data](https://github.com/junwang-wish/query_understanding_data), which documents how those data were created using [dvc](https://dvc.org/doc). 

Most likely [query_understand_data](https://github.com/junwang-wish/query_understanding_data) should have all data needed for model training. If more data is needed, please reach out to repo owner.
