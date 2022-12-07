# Product title embedding through contrastive learning

## [v0.0](version_0)
- Trained trained on {input: "iphone", output: "phone > apple"} with CLIP-style in-batch contrastive loss, such that embeddings of "iphone" is close to "phone > apple" in cosine distance
- Use [Aliexpress data](../../datasets/product_title_taxonomy_classification/wish-aliexpress.yaml) only
- To reproduce run below from `query_understanding_model` directory on 7 x V100-16GB instance within [Docker dev Container](../../.devcontainer/devcontainer.json)

`python main_embedding.py fit --config models/product_title_embedding/version_0/config.yaml`

## [v0.4](version_4)
- Trained trained on {input: "iphone", output: "phone > apple"} with CLIP-style in-batch contrastive loss, such that embeddings of "iphone" is close to "phone > apple" in partial-order distance
- Use [Aliexpress data](../../datasets/product_title_taxonomy_classification/wish-aliexpress.yaml) only
- To reproduce run below from `query_understanding_model` directory on 7 x V100-16GB instance within [Docker dev Container](../../.devcontainer/devcontainer.json)
- Training was interrupted multiple times, thus need to set `resume_from_checkpoint` to None in `config.yaml` if reproducing from scratch (saving all intermediate checkpoints to dvc / github is too messy). 

`python main_embedding.py fit --config models/product_title_embedding/version_4/config.yaml`
