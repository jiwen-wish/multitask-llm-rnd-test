import subprocess
import os 

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# configurable
model_relpath = "models/product_title_multitask_multimodal/version_1/epoch=0-step=75000.ckpt"
model_config_relpath = "models/product_title_multitask_multimodal/version_1/config.yaml"
model_task = "clm_singlemodal_wishtitle2pseudov121tax"
tax_constraint_relpath = "datasets/taxonomy/wish_v1.2.1_newtax_allpaths.txt"
num_beams = 3
num_return_sequences = 3
max_new_tokens = 50
output_dir_relpath = "models/product_title_multitask_multimodal/version_1"
data_config_relpath = "datasets/multimodal_multitask/wish_labelled_query_offshore_test.yaml"
data_source_type = "dvc"
model_name = "t5-base"
batch_size = 1
max_length = 50
max_length_out = 50
num_workers = 0
device = "gpu"
strategy = "ddp"

if __name__ == "__main__":
    strategy_args = [] if strategy is None else [ 
        '--trainer.strategy', strategy
    ]
    subprocess.run(
        [
            'python', 
            'main_inference_multimodal.py', 
            '--model', 'LLM_Inference_Multimodal', 
            '--model.llm_type', 'clm',
            '--model.output_scores', 'true',
            '--model.ckpt_path', model_relpath,
            '--model.config_path', model_config_relpath,
            '--model.task', model_task,
            '--model.allowed_gen_sequences', tax_constraint_relpath,
            '--model.num_beams', str(num_beams), 
            '--model.num_return_sequences', str(num_return_sequences), 
            '--model.do_sample', "false",
            '--model.length_penalty', str(0), # for proper sequence probs
            '--model.max_new_tokens', str(max_new_tokens),
            '--model.output_dir', output_dir_relpath,
            '--model.write_interval', 'batch',
            '--data', 'JSONListData',
            '--data.llm_type', 'clm',
            '--data.data_source_yaml_path', data_config_relpath, 
            '--data.input_dict', "{'template': '{query}', 'task_prefix': 'Generate taxonomy for query: '}",
            '--data.output_dict', "{'template': '{query}'}", 
            '--data.data_source_type', data_source_type,
            '--data.model_name', model_name,
            '--data.batch_size', str(batch_size),
            '--data.max_length', str(max_length),
            '--data.max_length_out', str(max_length_out),
            '--data.num_workers', str(num_workers),
            '--data.predict_on_test', 'true',
            '--trainer.logger', 'false', 
            '--trainer.enable_checkpointing', 'false',
            '--trainer.accelerator', device
        ] + strategy_args, 
        cwd=main_path
    )