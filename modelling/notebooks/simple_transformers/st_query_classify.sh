# depth 1
python st_query_classify.py --data_config ../../datasets/multimodal_multitask/wish_labelled_query_offshoreappen_and_pseudolabel_deduppathoverlap.yaml \
    --load_ckpt xlm-roberta-base --output_dir outputs_stage1 --truncate_depth 1
python st_query_classify.py --data_config ../../datasets/multimodal_multitask/wish_labelled_query_offshoreappen.yaml \
    --load_ckpt outputs_stage1/best_model --output_dir outputs_stage2 --truncate_depth 1
# depth 2
python st_query_classify.py --data_config ../../datasets/multimodal_multitask/wish_labelled_query_offshoreappen_and_pseudolabel_deduppathoverlap.yaml \
    --load_ckpt outputs_stage2/best_model --output_dir outputs_stage3 --truncate_depth 2
python st_query_classify.py --data_config ../../datasets/multimodal_multitask/wish_labelled_query_offshoreappen.yaml \
    --load_ckpt outputs_stage3/best_model --output_dir outputs_stage4 --truncate_depth 2
# depth 3
python st_query_classify.py --data_config ../../datasets/multimodal_multitask/wish_labelled_query_offshoreappen_and_pseudolabel_deduppathoverlap.yaml \
    --load_ckpt outputs_stage4/best_model --output_dir outputs_stage5 --truncate_depth 3
python st_query_classify.py --data_config ../../datasets/multimodal_multitask/wish_labelled_query_offshoreappen.yaml \
    --load_ckpt outputs_stage5/best_model --output_dir outputs_stage6 --truncate_depth 3
# depth 4
python st_query_classify.py --data_config ../../datasets/multimodal_multitask/wish_labelled_query_offshoreappen_and_pseudolabel_deduppathoverlap.yaml \
    --load_ckpt outputs_stage6/best_model --output_dir outputs_stage7 --truncate_depth 4
python st_query_classify.py --data_config ../../datasets/multimodal_multitask/wish_labelled_query_offshoreappen.yaml \
    --load_ckpt outputs_stage7/best_model --output_dir outputs_stage8 --truncate_depth 4
# depth 5
python st_query_classify.py --data_config ../../datasets/multimodal_multitask/wish_labelled_query_offshoreappen_and_pseudolabel_deduppathoverlap.yaml \
    --load_ckpt outputs_stage8/best_model --output_dir outputs_stage9 --truncate_depth 5
python st_query_classify.py --data_config ../../datasets/multimodal_multitask/wish_labelled_query_offshoreappen.yaml \
    --load_ckpt outputs_stage9/best_model --output_dir outputs_stage10 --truncate_depth 5