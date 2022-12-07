# Multitask Model

## [v0.1](version_1)
- [Slides](https://docs.google.com/presentation/d/1TGGTOUSLkOBpk_Eurzs0xKb_TFG70LvbnhgsHLQNdKc/edit?usp=share_link) 
- Trained on 
    -  clm_ali_title2tax (conditional LM loss on [Aliexpress title->v1.2.1taxonomy](../../datasets/product_title_taxonomy_classification/wish-aliexpress-simpleprompt-topdown.yaml)) using pairs of {input: "iphone", output: "phone > apple"}
    - dlm_wish_title (UL2-style denoising loss on [Wish title](../../datasets/product_title_denoise/wish-tahoe.yaml)) using pairs of {input: "iphone <extra_token_0>", output: "<extra_token_0>11<extra_token_1>"}
    - emb_all_title2tax (CLIP-style contrastive loss on [Wish,Aliexpress,Amazon title->different taxonomies](../../datasets/product_title_embedding/tahoe-aliexpress-amazon.yaml)) using pairs of {input: "iphone", output: "phone > apple"}
    - seqclf_ali_title2tax (Vanilla multilabel sequence classification with more weight on ancestor using [Aliexpress title->v1.2.1taxonomy](../../datasets/product_title_seqclf/wish-aliexpress.yaml)) using pairs of {input: "iphone", output: "phone > apple"}
- Model training was interrupted around ~10% of epoch 0 training, but still yields better performance than single task training on offshore-validated
- To reproduce run below from `query_understanding_model` directory on 6 x V100-16GB instance within [Docker dev Container](../../.devcontainer/devcontainer.json)

`python main_multitask.py fit --config models/product_title_multitask/version_1/config.yaml`