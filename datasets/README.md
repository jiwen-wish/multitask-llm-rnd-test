# query_understanding_data
Data repo for query understanding (query classification, etc.)

## High-level approach [Slide](https://docs.google.com/presentation/d/1W1jvTWK43Pxhx22XDuF7BuNuifnOEI_eLtQUeqDZraw/edit?usp=sharing), [Brainstorm](https://docs.google.com/document/d/1fsskXMGYeHsPGHaJYBrjus9Y7pDFdfG9cys2ehdqxQY/edit?usp=sharing)
- Simple approach based on language modelling
- Exploit 0-shot generalization to unseen domain due to pretraining
- Extensible and flexible to utilize newly available metadata (taxonomy) and modality (image, latent features)

## Important Notes
- Read [dvc doc](https://dvc.org/doc) before running any `dvc` commands in this repo or [query_understand_model](https://github.com/junwang-wish/query_understanding_model) repo. Otherwise, incorrect commands might wipe all trained models and data and are irreversible (not what we want).

## Reproduce

This repo should be run on your [Dev EC2 instance](https://wiki.wish.site/pages/viewpage.action?spaceKey=ENG&title=EC2+Instance+-+Dev+Env+setup) inside a [VScode Dev Docker Container](https://code.visualstudio.com/docs/remote/containers) to ensure full reproducibility.

### 1. Configure globalprotect VPN on EC2
You will need to be on Global Protect to query Tahoe data, which is part of the data pipeline. 
- Connect to Global Protect on your EC2 instance by running: ```globalprotect connect -p vpn-us.wish.site```
- Enter your okta USERNAME when prompted, which will be the same as your hostname
- Enter your okta password when promted
- Check DUO mobile app for a push notification and approve the notification to finish connecting to Global Protect
- Check global protect status using: globalprotect show --status

### 2. Have correct .bashrc and .ssh under $HOME directory

On your EC2 instance, have credentials needed to access AWS S3 and Tahoe in ```$HOME/.bashrc```, and Github access keys in ```$HOME/.ssh``` folder, since they will be mounted to the dev container to pass authentification variables:

```shell
junwang@junwang-ec2:~/query_understanding_data$ ls $HOME/.ssh

authorized_keys  id_ecdsa  id_ecdsa.pub  id_rsa  id_rsa.pub  known_hosts


junwang@junwang-ec2:~/query_understanding_data$ tail -n 20 $HOME/.bashrc

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
```

### 3. Reopen repo in Container using VScode
Dependencies needed to run data pipelines are installed during docker building.
- ```Shift-Command-P``` on Mac or ```Ctrl-Shift-P``` on Windows to open VScode menu
- Search and click ```Remote-Containers: Reopen in Container```
- Wait for dev docker build if it is first-time usage, otherwise existing built docker will be reused

### 4. Run dvc (Data Version Control) to reproduce all data pipelines
- Run ```dvc pull``` to download data stored in ```s3://structured-data-dev/junwang_query_understand_data/dvc```
- Run ```dvc repro``` which reproduces all data pipelines specified in ```dvc.yaml``` using hyperparameters specified in ```params.yaml```

### 5. Inspect data with notebooks
- Do EDA (Exploratory Data Analysis) with existing notebooks in ```notebooks/eda/*.ipynb``` or create new ones

### 6. Head to [query_understanding_model](https://github.com/junwang-wish/query_understanding_model)
- Understand how data is consumed for model training / evaluation
- Pick and choose dataset for new model experimentation
- When new data processing pipeline is needed, work on this repo
