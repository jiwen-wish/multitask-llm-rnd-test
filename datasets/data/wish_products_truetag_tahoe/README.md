# Wish product categorization data from Tahoe

## Manually Uploaded Data

- [TrueTag to Category Path Sheet](https://docs.google.com/spreadsheets/d/1lzp2BQFqV5CjyiY-ULsrL3tmyx0sHchLW-9yQdZSC3o/edit#gid=0) - [Active-Detail-Tags-2021.xlsx](Active-Detail-Tags-2021.xlsx.dvc) 
    -  This could be stale, use [Download Updated TrueTag Mapping Table](download_wish_truetag_info.py) instead for most up-to-date TrueTag mapping

## Code and DVC paths

- [Download Updated TrueTag Mapping Table](download_wish_truetag_info.py)
    - `data/wish_products_truetag_tahoe/true_tag_info.json`
- [Download Wish Tahoe Products Table](download_wish_truetag_tahoe_data.py)
    - `data/wish_products_truetag_tahoe/wishproducts_truetag_tahoe.json.gz`
- [Process Wish Tahoe Products Table](../../notebooks/preprocess/process_wish_truetag_tahoe_categories.py)
    - `data/wish_products_truetag_tahoe/Wish_Truetag_Tahoe_Meta_Train.json`
    - `data/wish_products_truetag_tahoe/Wish_Truetag_Tahoe_Meta_Val.json`
    - `data/wish_products_truetag_tahoe/Wish_Truetag_Tahoe_Meta_Test.json`
- [Infer Wish product taxonomy with few-shot OpenAI](../../notebooks/openai/few_shot_taxonomize_wish_tahoe.py)
    - `data/wish_products_truetag_tahoe/Wish_Truetag_Tahoe_Meta_Train_OpenaiInferred.json`
    - `data/wish_products_truetag_tahoe/Wish_Truetag_Tahoe_Meta_Val_OpenaiInferred.json`
    - `data/wish_products_truetag_tahoe/Wish_Truetag_Tahoe_Meta_Test_OpenaiInferred.json`
- [Process Wish OpenAI-inferred product taxonomy data](../../notebooks/preprocess/process_openai_inferred_wish_truetag_tahoe_categories.py)
    - `data/wish_products_truetag_tahoe/Wish_Truetag_Tahoe_Meta_Train_OpenaiInferred_Processed.json`
    - `data/wish_products_truetag_tahoe/Wish_Truetag_Tahoe_Meta_Val_OpenaiInferred_Processed.json`
    - `data/wish_products_truetag_tahoe/Wish_Truetag_Tahoe_Meta_Test_OpenaiInferred_Processed.json`
    