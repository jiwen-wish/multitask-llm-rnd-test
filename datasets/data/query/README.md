# Wish user query

## Manually Uploaded Data
- [Appen query classification task](https://docs.google.com/spreadsheets/d/1LLPtrmMQfcyPCm-b8QptnnYTY2COsgMIUeb44yc0qBo/edit?usp=sharing) - [Appen_Query_Understanding.xlsx](Appen_Query_Understanding.xlsx.dvc)

## Code and DVC Paths
- [Fetch top user search queries in a time range](top_used_wish_queries.py)
    - `data/query/top_used_wish_queries.json`
- [Infer user query new taxonomy based on log](https://github.com/ContextLogic/clroot/blob/77a1021c8234e41093074a59c9d5a8dc4093953a/sweeper/scripts/crons/search/query_new_category_inference_tahoe.py) - [infer_query_new_taxonomy.py](infer_query_new_taxonomy.py)
    - `data/query/wish_queries_inferred_newtax.json`
- [Process inferred user query new taxonomy based on log](../../notebooks/preprocess/process_inferred_wish_query_tax.py)
    - `data/query/Inferred_Wish_Query_Meta_Train.json`
    - `data/query/Inferred_Wish_Query_Meta_Val.json`
    - `data/query/Inferred_Wish_Query_Meta_Test.json`
    