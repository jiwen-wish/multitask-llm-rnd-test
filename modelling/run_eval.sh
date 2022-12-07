#!/bin/bash -v
set -x;
set -e;
# rm artifact
# rm -rf notebooks/eval/eval_aggregate.csv

# eval lance
# python notebooks/eval/eval_from_file.py -d mturk -v lance

# python notebooks/eval/eval_from_file.py -d offshore -v lance

python notebooks/eval/eval_from_file.py -d offshore-validated -v lance

# eval clm_v0.4
# python notebooks/eval/eval_from_file.py -d mturk \
#     -v clm_v0.4_aliprompt \
#     -i models/product_title_taxonomy_classification/version_4/epoch=0-step=996156--wish-mturk-test-aliprompt.json

# python notebooks/eval/eval_from_file.py -d mturk \
#     -v clm_v0.4_wishprompt \
#     -i models/product_title_taxonomy_classification/version_4/epoch=0-step=996156--wish-mturk-test-wishprompt.json

# python notebooks/eval/eval_from_file.py -d offshore \
#     -v clm_v0.4_aliprompt \
#     -i models/product_title_taxonomy_classification/version_4/epoch=0-step=996156--wish-offshore-test-aliprompt.json

# python notebooks/eval/eval_from_file.py -d offshore \
#     -v clm_v0.4_wishprompt \
#     -i models/product_title_taxonomy_classification/version_4/epoch=0-step=996156--wish-offshore-test-wishprompt.json

python notebooks/eval/eval_from_file.py -d offshore-validated \
    -v clm_v0.4_aliprompt \
    -i models/product_title_taxonomy_classification/version_4/epoch=0-step=996156--wish-offshore-validated-test-aliprompt.json

python notebooks/eval/eval_from_file.py -d offshore-validated \
    -v clm_v0.4_wishprompt \
    -i models/product_title_taxonomy_classification/version_4/epoch=0-step=996156--wish-offshore-validated-test-wishprompt.json

# eval clm_v0.5
# python notebooks/eval/eval_from_file.py -d mturk \
#     -v clm_v0.5_aliprompt \
#     -i models/product_title_taxonomy_classification/version_5/epoch=1-step=1006577--wish-mturk-test-aliprompt.json

# python notebooks/eval/eval_from_file.py -d mturk \
#     -v clm_v0.5_wishprompt \
#     -i models/product_title_taxonomy_classification/version_5/epoch=1-step=1006577--wish-mturk-test-wishprompt.json

# python notebooks/eval/eval_from_file.py -d offshore \
#     -v clm_v0.5_aliprompt \
#     -i models/product_title_taxonomy_classification/version_5/epoch=1-step=1006577--wish-offshore-test-aliprompt.json

# python notebooks/eval/eval_from_file.py -d offshore \
#     -v clm_v0.5_wishprompt \
#     -i models/product_title_taxonomy_classification/version_5/epoch=1-step=1006577--wish-offshore-test-wishprompt.json

python notebooks/eval/eval_from_file.py -d offshore-validated \
    -v clm_v0.5_aliprompt \
    -i models/product_title_taxonomy_classification/version_5/epoch=1-step=1006577--wish-offshore-validated-test-aliprompt.json

python notebooks/eval/eval_from_file.py -d offshore-validated \
    -v clm_v0.5_wishprompt \
    -i models/product_title_taxonomy_classification/version_5/epoch=1-step=1006577--wish-offshore-validated-test-wishprompt.json


# eval emb_v0.0
# python notebooks/eval/eval_from_file_emb.py --dataset mturk \
#     --input_title models/product_title_embedding/version_0/epoch=4-step=370604--wish-mturk-test.json \
#     --input_tax models/product_title_embedding/version_0/epoch=4-step=370604--wish-newtax-v1.2.1.json \
#     --version emb_v0.0 \
#     --distance_func cosine

# python notebooks/eval/eval_from_file_emb.py --dataset offshore \
#     --input_title models/product_title_embedding/version_0/epoch=4-step=370604--wish-offshore-test.json \
#     --input_tax models/product_title_embedding/version_0/epoch=4-step=370604--wish-newtax-v1.2.1.json \
#     --version emb_v0.0 \
#     --distance_func cosine

python notebooks/eval/eval_from_file_emb.py --dataset offshore-validated \
    --input_title models/product_title_embedding/version_0/epoch=4-step=370604--wish-offshore-validated-test.json \
    --input_tax models/product_title_embedding/version_0/epoch=4-step=370604--wish-newtax-v1.2.1.json \
    --version emb_v0.0 \
    --distance_func cosine

# eval emb_v0.4

python notebooks/eval/eval_from_file_emb.py --dataset offshore-validated \
    --input_title models/product_title_embedding/version_4/epoch=4-step=398293--wish-offshore-validated-test.json \
    --input_tax models/product_title_embedding/version_4/epoch=4-step=398293--wish-newtax-v1.2.1.json \
    --version emb_v0.4 \
    --distance_func order