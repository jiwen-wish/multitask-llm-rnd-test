# multitask-llm-rnd

- [modelling](./modelling/) - RnD modelling for multitask large language model (llm), primarily for query understanding but applies to content understanding tasks in general, migrated from [older modelling link](https://github.com/junwang-wish/query_understanding_model)
- [datasets](./datasets/) - Dataset creation for training and evaluating models, primarily for query understanding but applies to content understanding tasks in general, migrated from [older dataset link](https://github.com/junwang-wish/query_understanding_data)
- [labelling](./labelling/) - Labelling tools for collaborating with Wish offshore team to produce high-quality data in a secure fashion

## Dependency
### ~/.bashrc needs to have correct environment variables
Sample ~/.bashrc outside of any dev container
```
# other stuff
......

# add AWS
export AWS_ACCESS_KEY_ID=******
export AWS_SECRET_ACCESS_KEY=******
export AWS_KEY=******
export AWS_SECRET=******

# add Tahoe
export TAHOE_USERNAME=******
export TAHOE_PASSWORD=******

# add Gitlab
export GITLAB_USERNAME=******
export GITLAB_ACCESS_TOKEN=******

# add Huggingface cache dir
export HF_DATASETS_CACHE="/whateveryouwant"
export TRANSFORMERS_CACHE="/whateveryouwant2"

# add general cache dir
export GENERAL_CACHE="/whateveryouwant3"

# Add conda py38 path from dev container for modelling
export PATH="/opt/conda/envs/py38/bin:$PATH"

# Add prodigy key
export PRODIGY_KEY=******
export PRODIGY_HOME="/workspaces/multitask-llm-rnd/labelling/.prodigy" # this should be copied verbatim
export PRODIGY_HOST="0.0.0.0" # this should be copied verbatim
```

## Debug
### If there is permission error when adding certain files to Git
1. Check what usergroup you are in
```
groups $USER
```
2. Let's say you are in `developers` group from command above
```
cd .git
sudo chmod -R g+ws *
sudo chgrp -R developers *
git config core.sharedRepository true
```





