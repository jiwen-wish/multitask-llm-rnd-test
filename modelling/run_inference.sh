#!/bin/bash -v
set -x;
set -e;

# set hparams
DEVICES=3
ACCELERATOR="gpu"
CUDA_VISIBLE_DEVICES=0,1,2
STRATEGY="ddp"

# infer embedding for taxonomy
for DATA in $(ls datasets/taxonomy_embedding/*.yaml)
do
        # embedding lm inference v0.0
        python main_inference.py \
                --model=LLM_Inference_Embedding \
                --model.llm_type="emb" \
                --model.ckpt_path="models/product_title_embedding/version_0/epoch=4-step=370604.ckpt" \
                --model.config_path="models/product_title_embedding/version_0/config.yaml" \
                --model.output_dir="models/product_title_embedding/version_0" \
                --model.write_interval="batch" \
                --model.is_input_embedding=true \
                --data=LLM_EmbedData \
                --data.batch_size 16 \
                --data.data_source_yaml_path $DATA \
                --data.model_name="sentence-transformers/sentence-t5-large" \
                --data.overwrite_cache=false \
                --trainer.logger=false \
                --trainer.enable_checkpointing=false \
                --trainer.accelerator $ACCELERATOR \
                --trainer.devices $DEVICES \
                --trainer.strategy $STRATEGY
        sleep 10
        # embedding lm inference v0.4
        python main_inference.py \
                --model=LLM_Inference_Embedding \
                --model.llm_type="emb" \
                --model.ckpt_path="models/product_title_embedding/version_4/epoch=4-step=398293.ckpt" \
                --model.config_path="models/product_title_embedding/version_4/config.yaml" \
                --model.output_dir="models/product_title_embedding/version_4" \
                --model.write_interval="batch" \
                --model.is_input_embedding=true \
                --data=LLM_EmbedData \
                --data.batch_size 16 \
                --data.data_source_yaml_path $DATA \
                --data.model_name="sentence-transformers/sentence-t5-large" \
                --data.overwrite_cache=false \
                --trainer.logger=false \
                --trainer.enable_checkpointing=false \
                --trainer.accelerator $ACCELERATOR \
                --trainer.devices $DEVICES \
                --trainer.strategy $STRATEGY
        sleep 10
done

# # infer embedding for test set
# for DATA in $(ls datasets/product_title_embedding/*validated-test*.yaml)
# do
#         # embedding lm inference v0.0
#         python main_inference.py \
#                 --model=LLM_Inference_Embedding \
#                 --model.llm_type="emb" \
#                 --model.ckpt_path="models/product_title_embedding/version_0/epoch=4-step=370604.ckpt" \
#                 --model.config_path="models/product_title_embedding/version_0/config.yaml" \
#                 --model.output_dir="models/product_title_embedding/version_0" \
#                 --model.write_interval="batch" \
#                 --model.is_input_embedding=true \
#                 --data=LLM_EmbedData \
#                 --data.batch_size 16 \
#                 --data.data_source_yaml_path $DATA \
#                 --data.model_name="sentence-transformers/sentence-t5-large" \
#                 --data.overwrite_cache=false \
#                 --trainer.logger=false \
#                 --trainer.enable_checkpointing=false \
#                 --trainer.accelerator $ACCELERATOR \
#                 --trainer.devices $DEVICES \
#                 --trainer.strategy $STRATEGY
#         sleep 10
#         # embedding lm inference v0.4
#         python main_inference.py \
#                 --model=LLM_Inference_Embedding \
#                 --model.llm_type="emb" \
#                 --model.ckpt_path="models/product_title_embedding/version_4/epoch=4-step=398293.ckpt" \
#                 --model.config_path="models/product_title_embedding/version_4/config.yaml" \
#                 --model.output_dir="models/product_title_embedding/version_4" \
#                 --model.write_interval="batch" \
#                 --model.is_input_embedding=true \
#                 --data=LLM_EmbedData \
#                 --data.batch_size 16 \
#                 --data.data_source_yaml_path $DATA \
#                 --data.model_name="sentence-transformers/sentence-t5-large" \
#                 --data.overwrite_cache=false \
#                 --trainer.logger=false \
#                 --trainer.enable_checkpointing=false \
#                 --trainer.accelerator $ACCELERATOR \
#                 --trainer.devices $DEVICES \
#                 --trainer.strategy $STRATEGY
#         sleep 10
# done

# # infer lm
# for DATA in $(ls datasets/product_title_taxonomy_classification/*validated-test*.yaml)
# do
#         # conditional lm inference v0.4
#         python main_inference.py \
#                 --model LLM_Inference_Conditional_LM \
#                 --model.llm_type "clm" \
#                 --model.ckpt_path "models/product_title_taxonomy_classification/version_4/epoch=0-step=996156.ckpt" \
#                 --model.config_path "models/product_title_taxonomy_classification/version_4/config.yaml" \
#                 --model.output_dir="models/product_title_taxonomy_classification/version_4" \
#                 --model.write_interval="batch" \
#                 --model.allowed_gen_sequences "datasets/taxonomy/wish_v1.2.1_newtax_leafpaths.txt" \
#                 --data LLMData \
#                 --data.batch_size 16 \
#                 --data.data_source_yaml_path $DATA \
#                 --data.model_name "t5-large" \
#                 --data.overwrite_cache=false \
#                 --trainer.logger=false \
#                 --trainer.enable_checkpointing=false \
#                 --trainer.accelerator $ACCELERATOR \
#                 --trainer.devices $DEVICES \
#                 --trainer.strategy $STRATEGY
#         sleep 10
#         # conditional lm inference v0.5
#         python main_inference.py \
#                 --model LLM_Inference_Conditional_LM \
#                 --model.llm_type "clm" \
#                 --model.ckpt_path "models/product_title_taxonomy_classification/version_5/epoch=1-step=1006577.ckpt" \
#                 --model.config_path "models/product_title_taxonomy_classification/version_5/config.yaml" \
#                 --model.output_dir="models/product_title_taxonomy_classification/version_5" \
#                 --model.write_interval="batch" \
#                 --model.allowed_gen_sequences "datasets/taxonomy/wish_v1.2.1_newtax_leafpaths.txt" \
#                 --data LLMData \
#                 --data.batch_size 16 \
#                 --data.data_source_yaml_path $DATA \
#                 --data.model_name "t5-large" \
#                 --data.overwrite_cache=false \
#                 --trainer.logger=false \
#                 --trainer.enable_checkpointing=false \
#                 --trainer.accelerator $ACCELERATOR \
#                 --trainer.devices $DEVICES \
#                 --trainer.strategy $STRATEGY
#         sleep 10
# done 