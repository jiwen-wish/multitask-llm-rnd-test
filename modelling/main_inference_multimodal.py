#%%
import logging
logging.getLogger().setLevel(logging.INFO)

from typing import List, Union
import yaml
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from main_utils import Trie
from main_utils import (
    LLMData, LLM_EmbedData, LLM_SeqClassifyData, get_transformer
)
from main_utils_multimodal import JSONListData
import os
import torch
from main_inference import (PredictionWriter_Conditional_LM, PredictionWriter_SeqClassify, 
    PredictionWriter_Embedding, clean_exisiting_inference_artifacts, end_of_inference_logging
)
from main_multitask_multimodal import LLM_MultitaskMultimodal

class LLM_Inference_Multimodal(pl.LightningModule):
    def __init__(self, llm_type: str, ckpt_path: str, config_path: str, task: str,
            # clm
            num_beams: int=3, num_return_sequences: int=3, do_sample: bool=False, length_penalty: float=0, 
            max_new_tokens: int=50, allowed_gen_sequences: Union[List[str], str]=None, 
            output_scores: bool=False,
            # emb
            is_input_embedding: bool=True, 
            # multi modal
            use_multimodal_embedding: bool=False,
            # output
            output_dir: str=None, write_interval: str="batch", **kwargs
        ):
        super().__init__()
        assert llm_type in ['emb', 'clm', 'seqclf']
        assert task.startswith(llm_type)
        assert output_dir is not None
        self.save_hyperparameters()
        self.load_llm_ckpt(**kwargs)
        assert task in self.llm.hparams.multitask_specs_dict
        if llm_type == 'clm':
            buffer = []
            if allowed_gen_sequences is not None and isinstance(allowed_gen_sequences, str):
                with open(allowed_gen_sequences, 'r') as f:
                    for l in f:
                        l = l.strip()
                        if len(l):
                            buffer.append(l)
            
            allowed_gen_sequences = buffer
            self.load_trie(allowed_gen_sequences)
        

    def forward(self, batch_):
        use_embs = False
        if self.llm.hparams.multitask_specs_dict[self.hparams.task] is not None and \
                "multimodal_embedding" in self.llm.hparams.multitask_specs_dict[self.hparams.task] and \
                self.hparams.use_multimodal_embedding:
            batch = self.llm.transform_multimodal_batch(batch_, self.hparams.task)
            use_embs = True
        else:
            batch = batch_

        if self.hparams.llm_type == "clm":
            constraint = None
            if self.trie is not None:
                def constraint(batch_id, sent):
                    return self.trie.get(sent.tolist())
            if self.hparams.output_scores:
                if use_embs:
                    infres = self.llm.transformer.generate(
                        inputs_embeds = batch["inputs_embeds"], 
                        attention_mask = batch["attention_mask"],
                        num_beams = self.hparams.num_beams, 
                        num_return_sequences = self.hparams.num_return_sequences, 
                        do_sample = self.hparams.do_sample, 
                        length_penalty = self.hparams.length_penalty, 
                        max_new_tokens = self.hparams.max_new_tokens - 1, # HACK: T5 adds pad token in the beginning
                        prefix_allowed_tokens_fn=constraint, 
                        output_scores=True, return_dict_in_generate=True
                    )
                else:
                    infres = self.llm.transformer.generate(
                        input_ids = batch["input_ids"], 
                        attention_mask = batch["attention_mask"],
                        num_beams = self.hparams.num_beams, 
                        num_return_sequences = self.hparams.num_return_sequences, 
                        do_sample = self.hparams.do_sample, 
                        length_penalty = self.hparams.length_penalty, 
                        max_new_tokens = self.hparams.max_new_tokens - 1, # HACK: T5 adds pad token in the beginning
                        prefix_allowed_tokens_fn=constraint, 
                        output_scores=True, return_dict_in_generate=True
                    )
                prediction = infres.sequences
                probs = infres.sequences_scores.exp()
            else:
                if use_embs:
                    prediction = self.llm.transformer.generate(
                        inputs_embeds = batch["inputs_embeds"], 
                        attention_mask = batch["attention_mask"],
                        num_beams = self.hparams.num_beams, 
                        num_return_sequences = self.hparams.num_return_sequences, 
                        do_sample = self.hparams.do_sample, 
                        length_penalty = self.hparams.length_penalty, 
                        max_new_tokens = self.hparams.max_new_tokens - 1, # HACK: T5 adds pad token in the beginning
                        prefix_allowed_tokens_fn=constraint
                    )
                else:
                    prediction = self.llm.transformer.generate(
                        input_ids = batch["input_ids"], 
                        attention_mask = batch["attention_mask"],
                        num_beams = self.hparams.num_beams, 
                        num_return_sequences = self.hparams.num_return_sequences, 
                        do_sample = self.hparams.do_sample, 
                        length_penalty = self.hparams.length_penalty, 
                        max_new_tokens = self.hparams.max_new_tokens - 1, # HACK: T5 adds pad token in the beginning
                        prefix_allowed_tokens_fn=constraint
                    )

            # pad for multi-gpu to not have error
            if prediction.size(1) == self.hparams.max_new_tokens:
                prediction_padded = prediction
            elif prediction.size(1) < self.hparams.max_new_tokens:
                prediction_padded = torch.hstack((prediction, torch.full(
                        (prediction.size(0), self.hparams.max_new_tokens - prediction.size(1)), 
                        self.llm.tokenizer.pad_token_id, dtype=prediction.dtype
                    ).to(prediction.device)
                ))
            else:
                raise Exception((f"Prediction size {prediction.shape} exceed max_new_tokens "
                    f"on dim 1 {self.hparams.max_new_tokens}"))
            if self.hparams.output_scores:
                return prediction_padded, probs
            else:
                return prediction_padded
        
        elif self.hparams.llm_type == "emb":
            if self.hparams.is_input_embedding:
                batch_tmp = {i: batch[i] for i in batch if i in ["input_ids", "attention_mask", "inputs_embs"]}
            else:
                batch_tmp = {i.replace("output_", ""): batch[i] for i in batch if i in ["output_input_ids", "output_attention_mask", "output_inputs_embs"]}
            embs = self.llm.get_hidden_states(**batch_tmp)
            return embs
        
        elif self.hparams.llm_type == "seqclf":
            batch_tmp = {i: batch[i] for i in batch if i in ["input_ids", "attention_mask", "inputs_embs"]}
            hidden_states = self.llm.get_hidden_states(**batch_tmp)
            head = self.llm.hparams.multitask_specs_dict[self.hparams.task]["specs"]["clf_head"]
            logits = self.llm.get_submodule(head)(hidden_states)
            return logits
        
        else:
            raise NotImplemented()

    def load_llm_ckpt(self, **kwargs):
        llm_config = yaml.safe_load(open(self.hparams.config_path, 'r'))
        tmp_path = self.hparams.ckpt_path
        if llm_config['trainer']['strategy'] is not None and \
            'deepspeed' in llm_config['trainer']['strategy']:
            tmp_path = os.path.join(self.hparams.ckpt_path, 'pytorch_model.bin')
            if not os.path.exists(tmp_path):
                logging.info(f"Creating {tmp_path} from deepspeed shards")
                convert_zero_checkpoint_to_fp32_state_dict(self.hparams.ckpt_path, tmp_path)
            else:
                logging.info(f"Use already converted ckpt {tmp_path} from deepspeed shards")

        self.llm = LLM_MultitaskMultimodal.load_from_checkpoint(tmp_path, strict=False, **kwargs)

    def load_trie(self, allowed_gen_sequences):
        self.hparams.allowed_gen_sequences = allowed_gen_sequences
        if allowed_gen_sequences is None or len(allowed_gen_sequences) == 0:
            self.trie = None
        else:
            self.trie = Trie([
                [self.llm.tokenizer.pad_token_id] + self.llm.tokenizer.encode(i) + \
                    [self.llm.tokenizer.eos_token_id] for i in allowed_gen_sequences
            ])
    
    def predict_step(self, batch, batch_idx):
        return self.forward(batch)
    

#%%
def cli_main():
    """LightningCLI handles CLI with no boilerplate:
    Please read https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli_advanced_3.html?highlight=class_path#models-with-multiple-submodules 

    1. Example command for conditional LM task

    # no output sequence probabilities 

    python main_inference_multimodal.py \
        --model=LLM_Inference_Multimodal \
        --model.llm_type="clm" \
        --model.ckpt_path="models/product_title_multitask_multimodal/version_1/epoch=0-step=75000.ckpt" \
        --model.config_path="models/product_title_multitask_multimodal/version_1/config.yaml" \
        --model.task="clm_singlemodal_wishtitle2pseudov121tax" \
        --model.allowed_gen_sequences="datasets/taxonomy/wish_v1.2.1_newtax_leafpaths.txt" \
        --model.output_dir="models/product_title_multitask_multimodal/version_1" \
        --model.write_interval="batch" \
        --data=JSONListData \
        --data.llm_type="clm" \
        --data.data_source_yaml_path="datasets/multimodal_multitask/wish_offshore_validated.yaml" \
        --data.input_dict="{'template': '{title}', 'task_prefix': 'Generate taxonomy for product: '}" \
        --data.output_dict="{'template': '{category}'}" \
        --data.transform_dict="{'category': 'taxonomy'}" \
        --data.data_source_type="dvc" \
        --data.model_name="t5-base" \
        --data.batch_size=50 \
        --data.max_length=50 \
        --data.num_workers=0 \
        --trainer.logger=false \
        --trainer.enable_checkpointing=false \
        --trainer.accelerator gpu \
        --trainer.strategy ddp

    # output sequence probabilities

    python main_inference_multimodal.py \
        --model=LLM_Inference_Multimodal \
        --model.llm_type="clm" \
        --model.output_scores=true \
        --model.ckpt_path="models/product_title_multitask_multimodal/version_1/epoch=0-step=75000.ckpt" \
        --model.config_path="models/product_title_multitask_multimodal/version_1/config.yaml" \
        --model.task="clm_singlemodal_wishtitle2pseudov121tax" \
        --model.allowed_gen_sequences="datasets/taxonomy/wish_v1.2.1_newtax_leafpaths.txt" \
        --model.output_dir="models/product_title_multitask_multimodal/version_1" \
        --model.write_interval="batch" \
        --data=JSONListData \
        --data.llm_type="clm" \
        --data.data_source_yaml_path="datasets/multimodal_multitask/wish_joinv2_queries_en_20221130140116.yaml" \
        --data.input_dict="{'template': '{query}', 'task_prefix': 'Generate taxonomy for query: '}" \
        --data.output_dict="{'template': '{query}'}" \
        --data.data_source_type="dvc" \
        --data.model_name="t5-base" \
        --data.batch_size=50 \
        --data.max_length=50 \
        --data.num_workers=0 \
        --trainer.logger=false \
        --trainer.enable_checkpointing=false \
        --trainer.accelerator gpu \
        --trainer.strategy ddp
    
    2. Example command for seqclf task

    python main_inference_multimodal.py \
        --model=LLM_Inference_Multimodal \
        --model.llm_type="seqclf" \
        --model.ckpt_path="models/product_title_multitask_multimodal/version_1/epoch=0-step=75000.ckpt" \
        --model.config_path="models/product_title_multitask_multimodal/version_1/config.yaml" \
        --model.task="seqclf_singlemodal_wishtitle2pseudov121tax" \
        --model.output_dir="models/product_title_multitask_multimodal/version_1" \
        --model.write_interval="batch" \
        --data=JSONListData \
        --data.llm_type="seqclf" \
        --data.label_map_file="datasets/taxonomy/wish_v1.2.1_newtax_allpaths.txt" \
        --data.label_type="taxonomy" \
        --data.data_source_yaml_path="datasets/multimodal_multitask/wish_offshore_validated.yaml" \
        --data.input_dict="{'template': '{title}', 'task_prefix': 'Classify product: '}" \
        --data.output_dict="{'template': '{category}'}" \
        --data.transform_dict="{'category': 'taxonomy'}" \
        --data.data_source_type="dvc" \
        --data.model_name="t5-base" \
        --data.batch_size=50 \
        --data.max_length=50 \
        --data.num_workers=0 \
        --trainer.logger=false \
        --trainer.enable_checkpointing=false \
        --trainer.accelerator gpu \
        --trainer.strategy ddp

    3. Example command for emb task
    
    # embed title
    python main_inference_multimodal.py \
        --model=LLM_Inference_Multimodal \
        --model.llm_type="emb" \
        --model.ckpt_path="models/product_title_multitask_multimodal/version_1/epoch=0-step=75000.ckpt" \
        --model.config_path="models/product_title_multitask_multimodal/version_1/config.yaml" \
        --model.task="emb_singlemodal_wishtitle2pseudov121tax" \
        --model.output_dir="models/product_title_multitask_multimodal/version_1" \
        --model.write_interval="batch" \
        --data=JSONListData \
        --data.llm_type="emb" \
        --data.data_source_yaml_path="datasets/multimodal_multitask/wish_offshore_validated.yaml" \
        --data.input_dict="{'template': '{title}', 'task_prefix': 'Embed product: '}" \
        --data.output_dict="{'template': '{category}', 'task_prefix': 'Embed taxonomy: '}" \
        --data.transform_dict="{'category': 'taxonomy'}" \
        --data.data_source_type="dvc" \
        --data.model_name="t5-base" \
        --data.batch_size=50 \
        --data.max_length=50 \
        --data.num_workers=0 \
        --trainer.logger=false \
        --trainer.enable_checkpointing=false \
        --trainer.accelerator gpu \
        --trainer.strategy ddp

    # embed taxonomy
    python main_inference_multimodal.py \
        --model=LLM_Inference_Multimodal \
        --model.llm_type="emb" \
        --model.is_input_embedding=false \
        --model.ckpt_path="models/product_title_multitask_multimodal/version_1/epoch=0-step=75000.ckpt" \
        --model.config_path="models/product_title_multitask_multimodal/version_1/config.yaml" \
        --model.task="emb_singlemodal_wishtitle2pseudov121tax" \
        --model.output_dir="models/product_title_multitask_multimodal/version_1" \
        --model.write_interval="batch" \
        --data=JSONListData \
        --data.llm_type="emb" \
        --data.data_source_yaml_path="datasets/multimodal_multitask/wish-newtax-v1.2.1.yaml" \
        --data.input_dict="{'template': '{title}', 'task_prefix': 'Embed product: '}" \
        --data.output_dict="{'template': '{category}', 'task_prefix': 'Embed taxonomy: '}" \
        --data.transform_dict="{'category': 'taxonomy'}" \
        --data.data_source_type="dvc" \
        --data.model_name="t5-base" \
        --data.batch_size=50 \
        --data.max_length=50 \
        --data.num_workers=0 \
        --trainer.logger=false \
        --trainer.enable_checkpointing=false \
        --trainer.accelerator gpu \
        --trainer.strategy ddp

    4. Example command for multimodal conditional LM task 

    # Notice the task is "clm_multimodal_clip2wishtitle" instead of "*_2label" because multimodal
    # clm of taxonomy is not in our training, we rely on 0-shot task transfer

    python main_inference_multimodal.py \
        --model=LLM_Inference_Multimodal \
        --model.llm_type="clm" \
        --model.ckpt_path="models/product_title_multitask_multimodal/version_1/epoch=0-step=75000.ckpt" \
        --model.config_path="models/product_title_multitask_multimodal/version_1/config.yaml" \
        --model.task="clm_multimodal_clip2wishtitle" \
        --model.allowed_gen_sequences="datasets/taxonomy/wish_v1.2.1_newtax_leafpaths.txt" \
        --model.output_dir="models/product_title_multitask_multimodal/version_1" \
        --model.use_multimodal_embedding=true \
        --model.write_interval="batch" \
        --data=JSONListData \
        --data.llm_type="clm" \
        --data.data_source_yaml_path="datasets/multimodal_multitask/wish_offshore_validated_wclip.yaml" \
        --data.input_dict="{'template': '[title start] {title} [title end] [image start] {img_embedding} [image end]', \
            'task_prefix': 'Generate taxonomy for product with image: ', 'is_multimodal_embedding': ['img_embedding']}" \
        --data.output_dict="{'template': '{category}'}" \
        --data.transform_dict="{'category': 'taxonomy'}" \
        --data.data_source_type="dvc" \
        --data.model_name="t5-base" \
        --data.batch_size=50 \
        --data.max_length=50 \
        --data.num_workers=0 \
        --trainer.logger=false \
        --trainer.enable_checkpointing=false \
        --trainer.accelerator gpu \
        --trainer.strategy ddp

    5. Example command for multimodal seqclf task

    python main_inference_multimodal.py \
        --model=LLM_Inference_Multimodal \
        --model.llm_type="seqclf" \
        --model.ckpt_path="models/product_title_multitask_multimodal/version_1/epoch=0-step=75000.ckpt" \
        --model.config_path="models/product_title_multitask_multimodal/version_1/config.yaml" \
        --model.task="seqclf_multimodal_wishtitlewclip2pseudov121tax" \
        --model.output_dir="models/product_title_multitask_multimodal/version_1" \
        --model.use_multimodal_embedding=true \
        --model.write_interval="batch" \
        --data=JSONListData \
        --data.llm_type="seqclf" \
        --data.label_map_file="datasets/taxonomy/wish_v1.2.1_newtax_allpaths.txt" \
        --data.label_type="taxonomy" \
        --data.data_source_yaml_path="datasets/multimodal_multitask/wish_offshore_validated_wclip.yaml" \
        --data.input_dict="{'template': '[title start] {title} [title end] [image start] {img_embedding} [image end]', \
            'task_prefix': 'Classify product with image: ', 'is_multimodal_embedding': ['img_embedding']}" \
        --data.output_dict="{'template': '{category}'}" \
        --data.transform_dict="{'category': 'taxonomy'}" \
        --data.data_source_type="dvc" \
        --data.model_name="t5-base" \
        --data.batch_size=50 \
        --data.max_length=50 \
        --data.num_workers=0 \
        --trainer.logger=false \
        --trainer.enable_checkpointing=false \
        --trainer.accelerator gpu \
        --trainer.strategy ddp

    6. Example command for multimodal emb task

    # taxonomy doesn't have image, so reuse single modal emb inference of taxonomy embedding

    # embed title
    python main_inference_multimodal.py \
        --model=LLM_Inference_Multimodal \
        --model.llm_type="emb" \
        --model.ckpt_path="models/product_title_multitask_multimodal/version_1/epoch=0-step=75000.ckpt" \
        --model.config_path="models/product_title_multitask_multimodal/version_1/config.yaml" \
        --model.task="emb_singlemodal_wishtitle2pseudov121tax" \
        --model.output_dir="models/product_title_multitask_multimodal/version_1" \
        --model.use_multimodal_embedding=true \
        --model.write_interval="batch" \
        --data=JSONListData \
        --data.llm_type="emb" \
        --data.data_source_yaml_path="datasets/multimodal_multitask/wish_offshore_validated_wclip.yaml" \
        --data.input_dict="{'template': '[title start] {title} [title end] [image start] {img_embedding} [image end]', \
            'task_prefix': 'Embed product with image: ', 'is_multimodal_embedding': ['img_embedding']}" \
        --data.output_dict="{'template': '{category}', 'task_prefix': 'Embed taxonomy: '}" \
        --data.transform_dict="{'category': 'taxonomy'}" \
        --data.data_source_type="dvc" \
        --data.model_name="t5-base" \
        --data.batch_size=50 \
        --data.max_length=50 \
        --data.num_workers=0 \
        --trainer.logger=false \
        --trainer.enable_checkpointing=false \
        --trainer.accelerator gpu \
        --trainer.strategy ddp

    """
    cli = LightningCLI(run=False, save_config_callback=None)
    transformer_config, _ = get_transformer(cli.config.data.init_args.model_name, return_model=False)
    fname_prefix = (cli.model.hparams.ckpt_path.split('/')[-1].replace('.ckpt', '') + '--' + 
                cli.datamodule.hparams.data_source_yaml_path.split('/')[-1].replace('.yaml', ''))  
    
    if cli.config.data.init_args.predict_on_test:
        split = "test"
    elif cli.config.data.init_args.predict_on_trainval is not None:
        split = cli.config.data.init_args.predict_on_trainval
    else:
        split = "predict"

    if cli.config.model.init_args.use_multimodal_embedding:
        fname_prefix = fname_prefix + '--multimodal'

    if cli.config.model.init_args.llm_type.startswith('emb'):
        if cli.config.model.init_args.is_input_embedding:
            fname_prefix = fname_prefix + '--inputemb'
        else:
            fname_prefix = fname_prefix + '--outputemb'
    
    fname_prefix = fname_prefix + f'--{split}'
    
    fname_prefix = cli.config.model.init_args.llm_type + '-' + fname_prefix
    clean_exisiting_inference_artifacts(cli.config.model.init_args.output_dir, fname_prefix)

    if cli.config.model.init_args.llm_type.startswith('clm'):
        write_callback = PredictionWriter_Conditional_LM(
            output_dir=cli.config.model.init_args.output_dir,
            write_interval=cli.config.model.init_args.write_interval,
            fname_prefix=fname_prefix, 
            max_new_tokens=cli.config.model.init_args.max_new_tokens,
            num_return_sequences=cli.config.model.init_args.num_return_sequences,
            output_scores=cli.config.model.init_args.output_scores
        )
    elif cli.config.model.init_args.llm_type.startswith('emb'):
        if transformer_config.is_encoder_decoder:
            emb_dim = transformer_config.d_model
        else:
            emb_dim = transformer_config.hidden_size
        write_callback = PredictionWriter_Embedding(
            output_dir=cli.config.model.init_args.output_dir,
            write_interval=cli.config.model.init_args.write_interval,
            fname_prefix=fname_prefix,
            emb_dim=emb_dim
        )
    elif cli.config.model.init_args.llm_type.startswith('seqclf'):
        write_callback = PredictionWriter_SeqClassify(
            output_dir=cli.config.model.init_args.output_dir,
            write_interval=cli.config.model.init_args.write_interval,
            fname_prefix=fname_prefix, 
            label_map_file=cli.config.data.init_args.label_map_file
        )
    else:
        raise NotImplemented()
    logging.info(f"write_callback created")
    cli.trainer.callbacks.append(write_callback)
    cli.trainer.predict(cli.model, cli.datamodule, return_predictions=False)
    end_of_inference_logging(cli.config.model.init_args.output_dir, fname_prefix)
    

if __name__ == "__main__":
    cli_main()