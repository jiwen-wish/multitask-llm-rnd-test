# Multitask Multimodal Model

## [v0.1](version_1)
- [Slides](https://docs.google.com/presentation/d/1G6rGJQzMh6D3IrHGNSfy31SQ4ysqRDgcAQisu-8axDs/edit?usp=share_link) 
- To reproduce run below from `query_understanding_model` directory on 5 x V100-16GB instance within [Docker dev Container](../../.devcontainer/devcontainer.json). Note we stopped training at 75000 steps by eyeballing tensorboard, and the stopping is due to pragmatic OKR timeline reason (we cannot wait forever) and nothing else.

`python main_multitask_multimodal.py fit --config models/product_title_multitask_multimodal/version_1/config.yaml`