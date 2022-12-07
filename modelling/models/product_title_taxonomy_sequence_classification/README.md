# Product title taxonomy classification through sequence classification

## Summary

Focal loss and label_weight (whether weigh ancestor more / less / same as leaf labels) don't have meaningful impact on support-weighted F1 on L1 on [offshore-validated](../../datasets/product_title_seqclf/wish-offshore-validated-test.yaml) as shown in [eval](../../notebooks/eval/README.md).

Counter-intuitively, [hierarchical logit constraint](https://arxiv.org/abs/2203.14335) dramatically worsens performance (F1 drops to 0), either due to my implementation having a bug or other reasons. However the train and val loss during training smoothly does down still. Investigating the root cause for it is deprioritized as it would likely lead to insignificant marginal gain.

For future incorporation of seqclf loss into multitask mixture, for simplicity, we will weight ancestor label more than leaf, since [v0.7](version_7) has marginal gain over other variants, and use no additional tricks (focal loss / hierarchical logit constraint).

## [v0.0](version_0)
- Trained trained on {input: “iphone”, output: “phone > apple”} by hardcoding "phone > apple" into y("phone") = 1, y("phone > apple") = 1, y("food") = 0, etc. as multilabel problem
- Weight ancestors less than leaf labels (why? just experimenting, no reason), and train with Binary Cross Entropy loss
- Use [Aliexpress data](../../datasets/product_title_taxonomy_classification/wish-aliexpress.yaml) only
- To reproduce run below from `query_understanding_model` directory on 7 x V100-16GB instance within [Docker dev Container](../../.devcontainer/devcontainer.json)

`python main_seqclassify.py fit --config models/product_title_taxonomy_sequence_classification/version_0/config.yaml`

## [v0.2](version_2)
- Added focal loss trick compared to [v0.0](version_0)
- Everything else is the same as [v0.0](version_0)
- To reproduce run below from `query_understanding_model` directory on 7 x V100-16GB instance within [Docker dev Container](../../.devcontainer/devcontainer.json)

`python main_seqclassify.py fit --config models/product_title_taxonomy_sequence_classification/version_2/config.yaml`

## [v0.6](version_6)
- Weight ancestors same as leaf labels compared to [v0.0](version_0)
- Everything else is the same as [v0.0](version_0)
- To reproduce run below from `query_understanding_model` directory on 7 x V100-16GB instance within [Docker dev Container](../../.devcontainer/devcontainer.json)

`python main_seqclassify.py fit --config models/product_title_taxonomy_sequence_classification/version_6/config.yaml`

## [v0.7](version_7)
- Weight ancestors more than leaf labels compared to [v0.0](version_0)
- Everything else is the same as [v0.0](version_0)
- To reproduce run below from `query_understanding_model` directory on 7 x V100-16GB instance within [Docker dev Container](../../.devcontainer/devcontainer.json)

`python main_seqclassify.py fit --config models/product_title_taxonomy_sequence_classification/version_7/config.yaml`

## [v0.8](version_8)
- Add focal loss compared to [v0.7](version_7)
- Everything else is the same as [v0.7](version_7)
- To reproduce run below from `query_understanding_model` directory on 7 x V100-16GB instance within [Docker dev Container](../../.devcontainer/devcontainer.json)

`python main_seqclassify.py fit --config models/product_title_taxonomy_sequence_classification/version_8/config.yaml`

## [v0.9](version_9)
- Add focal loss compared to [v0.6](version_6)
- Everything else is the same as [v0.6](version_6)
- To reproduce run below from `query_understanding_model` directory on 7 x V100-16GB instance within [Docker dev Container](../../.devcontainer/devcontainer.json)

`python main_seqclassify.py fit --config models/product_title_taxonomy_sequence_classification/version_9/config.yaml`