# demo conditional lm model
rm -rf models/demo* 
python main_conditional_lm.py fit --config demo_config_conditional_lm.yaml
python main_denoise_lm.py fit --config demo_config_denoise_lm.yaml
python main_embedding.py fit --config demo_config_embedding.yaml
python main_seqclassify.py fit --config demo_config_seqclassify.yaml
python main_multitask.py fit --config demo_config_multitask.yaml