# Wish product categorization data (gotcha: actually data is from aliexpress)

## Manually Uploaded Data

- [Wish product Mturk test set 09202022](https://docs.google.com/spreadsheets/d/1Vsvxa9KXLkdPrmDRB5inzHh388b5YItdKHm--4OgwJA/edit?usp=sharing) - [wish-mturk-labelled-09202022.xlsx](wish-mturk-labelled-09202022.xlsx.dvc)
    - Usable Mturk labelled data as test set for wish product listing model

## Code and DVC paths 
- [Wish (Aliexpress) product categorization data](https://github.com/ContextLogic/listing-inference-training/blob/master/training/wish_categories/prepare_data.py) - [download_wish_products_data.py](download_wish_products_data.py)
    - Script to produce pickle file of (train, val, category) dataframs
    - Modified from [listing-inference-training](https://github.com/ContextLogic/listing-inference-training/blob/master/training/wish_categories/prepare_data.py)
    - DVC path: `data/wish_products/wishproducts.pkl`

- [Process Wish (Aliexpress) product categorization data](../../notebooks/preprocess/process_wish_product_categories.py)
    - `data/wish_products/Wish_Meta_Train.json`
    - `data/wish_products/Wish_Meta_Val.json`
    - `data/wish_products/Wish_Meta_Test.json`
    
- [Process Wish product Mturk test set 09202022](../../notebooks/preprocess/process_wish_mturk_labelled_09202022.py)
    - `data/wish_products/wish-mturk-labelled-09202022-clean.json`

- [Join processed Wish product Mturk test set 09202022 with Lance model results](../../notebooks/preprocess/join_lance_model_pred_wish_mturk_labelled_09202022.py)
    - `data/wish_products/wish-mturk-labelled-09202022-clean-joinedlance.json`

