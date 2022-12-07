#%%
import logging
logging.getLogger().setLevel(logging.INFO)

from typing import List, Any, Union
import yaml
import numpy as np
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.utilities import rank_zero_only
from main_conditional_lm import LLM 
from main_embedding import LLM_Embed
from main_seqclassify import LLM_SeqClassify
from main_utils import (
    Trie, get_transformer, LLM_EmbedData, 
    LLM_SeqClassifyData, LLMData, LLM_EmbedManualData, 
    LLM_SeqClassifyInputOnlyData
)
import os
import zarr
import torch
import shutil
import json
import pathlib

class PredictionWriter_Conditional_LM(BasePredictionWriter):
    def __init__(self, output_dir: str, write_interval: str, fname_prefix: str, 
            max_new_tokens: int, num_return_sequences: int = 1, output_scores: bool=False):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.fname_prefix = fname_prefix
        self.max_new_tokens = max_new_tokens
        self.num_return_sequences = num_return_sequences
        self.output_scores = output_scores
        self.zarr_handle = zarr.open(
            os.path.join(self.output_dir, self.fname_prefix + '.zarr'), 'a'
        )
        self.json_handle = open(
            os.path.join(self.output_dir, self.fname_prefix + '.json'), 'a'
        )

    def write_on_batch_end(
        self, trainer, pl_module: pl.LightningModule, prediction: Any, 
        batch_indices: List[int], batch: Any,
        batch_idx: int, dataloader_idx: int
    ):  
        if self.output_scores:
            prediction, probs = prediction
            probs = probs.detach().cpu().numpy()
            prediction = prediction.detach().cpu()
        else:
            prediction = prediction.detach().cpu()
        rank_indices = np.tile(np.arange(self.num_return_sequences), len(batch_indices)).tolist()
        batch_indices = np.repeat(batch_indices, self.num_return_sequences).tolist()
        assert prediction.size(0) == len(batch_indices) == len(rank_indices)
        
        prediction_batch_idx = torch.full(
            (prediction.size(0),1), batch_idx, dtype=prediction.dtype)
        prediction_dataloader_idx = torch.full(
            (prediction.size(0),1), dataloader_idx, dtype=prediction.dtype)
        prediction_decoded = pl_module.llm.tokenizer.batch_decode(prediction, skip_special_tokens=True)
        
        if prediction.size(1) == pl_module.hparams.max_new_tokens:
            prediction_padded = prediction
        elif prediction.size(1) < pl_module.hparams.max_new_tokens:
            prediction_padded = torch.hstack((prediction, torch.full(
                    (prediction.size(0), pl_module.hparams.max_new_tokens - prediction.size(1)), 
                    pl_module.llm.tokenizer.pad_token_id, dtype=prediction.dtype
                )
            ))
        else:
            raise Exception((f"Prediction size {prediction.shape} exceed max_new_tokens "
                f"on dim 1 {pl_module.hparams.max_new_tokens}"))

        # merge arrs
        out_arr = np.hstack((
            prediction_dataloader_idx.numpy(),
            prediction_batch_idx.numpy(),
            np.array(batch_indices).reshape(-1,1),
            np.array(rank_indices).reshape(-1,1),
            prediction_padded.numpy()
        ))

        global_rank = pl_module.global_rank
        
        if f'dl-idx_b-idx_s-idx_r-idx_pred_rank-{global_rank}' not in self.zarr_handle:
            self.zarr_handle.require_dataset(f'dl-idx_b-idx_s-idx_r-idx_pred_rank-{global_rank}', 
                # prepend 4 integers:
                # - dataloader_idx, 
                # - batch_idx, 
                # - batch_indices (sample index),
                # - rank_indices 
                # to prediction to avoid parallel write messing up with ordering
                shape=(0, 4 + self.max_new_tokens), 
                chunks=(10000, 4 + self.max_new_tokens), dtype='int')

        self.zarr_handle[f'dl-idx_b-idx_s-idx_r-idx_pred_rank-{global_rank}'].append(out_arr)

        for ind, i in enumerate(prediction_decoded):
            if self.output_scores:
                self.json_handle.write(json.dumps({
                    'prediction_decoded': i,
                    'prob': float(probs[ind]),
                    'batch_idx': batch_idx, 
                    'dataloader_idx': dataloader_idx, 
                    'batch_indices': batch_indices[ind],
                    'rank_indices': rank_indices[ind]
                }) + '\n')
            else:
                self.json_handle.write(json.dumps({
                    'prediction_decoded': i,
                    'batch_idx': batch_idx, 
                    'dataloader_idx': dataloader_idx, 
                    'batch_indices': batch_indices[ind],
                    'rank_indices': rank_indices[ind]
                }) + '\n')


class PredictionWriter_SeqClassify(BasePredictionWriter):
    def __init__(self, output_dir: str, write_interval: str, fname_prefix: str, 
            label_map_file: str):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.fname_prefix = fname_prefix
        self.label_map_file = label_map_file
        self.label_map = {} # label text to index
        with open(self.label_map_file, 'r') as f:
            for l in f:
                l = l.replace('\n', '').strip()
                if len(l):
                    self.label_map[l] = len(self.label_map)
        self.label_map_rev = {}
        for i in self.label_map:
            j = self.label_map[i]
            self.label_map_rev[j] = i 

        self.zarr_handle = zarr.open(
            os.path.join(self.output_dir, self.fname_prefix + '.zarr'), 'a'
        )
        self.json_handle = open(
            os.path.join(self.output_dir, self.fname_prefix + '.json'), 'a'
        )
    
    def write_on_batch_end(
        self, trainer, pl_module: pl.LightningModule, prediction: Any, 
        batch_indices: List[int], batch: Any,
        batch_idx: int, dataloader_idx: int
    ):
        probs = prediction.sigmoid()
        top_probs, top_pred_indices = probs.topk(min(probs.size(1), 10))
        assert prediction.size(1) == len(self.label_map)
        
        prediction_batch_idx = torch.full(
            (prediction.size(0),1), batch_idx, dtype=prediction.dtype)
        prediction_dataloader_idx = torch.full(
            (prediction.size(0),1), dataloader_idx, dtype=prediction.dtype)

        prediction_np = prediction.detach().cpu().numpy()
        
        # combine arrs
        out_arr = np.hstack((
            prediction_dataloader_idx.numpy(),
            prediction_batch_idx.numpy(),
            np.array(batch_indices).reshape(-1,1),
            prediction_np
        ))
        
        global_rank = pl_module.global_rank
        
        if f'dl-idx_b-idx_s-idx_pred_rank-{global_rank}' not in self.zarr_handle:
            self.zarr_handle.require_dataset(f'dl-idx_b-idx_s-idx_pred_rank-{global_rank}', 
                # prepend 3 integers (casted to float):
                # - dataloader_idx, 
                # - batch_idx, 
                # - batch_indices (sample index),
                # to prediction to avoid parallel write messing up with ordering
                shape=(0, 3 + len(self.label_map)), 
                chunks=(10000, 3 + len(self.label_map)), dtype='float')

        self.zarr_handle[f'dl-idx_b-idx_s-idx_pred_rank-{global_rank}'].append(out_arr)

        top_pred_indices = top_pred_indices.detach().cpu().numpy()
        top_probs = top_probs.detach().cpu().numpy()
        logits = prediction.detach().cpu().numpy()
        for ind, i in enumerate(batch_indices):
            self.json_handle.write(json.dumps({
                'preds': [(
                    self.label_map_rev[top_pred_indices[ind][ind_p].item()], 
                    p.item()
                ) for ind_p, p in enumerate(top_probs[ind])],
                'batch_idx': batch_idx, 
                'dataloader_idx': dataloader_idx, 
                'batch_indices': i,
                'logits': logits[ind].tolist()
            }) + '\n')
    


class PredictionWriter_Embedding(BasePredictionWriter):
    def __init__(self, output_dir: str, write_interval: str, fname_prefix: str, 
            emb_dim: int):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.fname_prefix = fname_prefix
        self.emb_dim = emb_dim
        self.zarr_handle = zarr.open(
            os.path.join(self.output_dir, self.fname_prefix + '.zarr'), 'a'
        )
        self.json_handle = open(
            os.path.join(self.output_dir, self.fname_prefix + '.json'), 'a'
        )
    
    def write_on_batch_end(
        self, trainer, pl_module: pl.LightningModule, prediction: Any, 
        batch_indices: List[int], batch: Any,
        batch_idx: int, dataloader_idx: int
    ):
        prediction = prediction.detach().cpu()
        assert prediction.size(1) == self.emb_dim
        
        prediction_batch_idx = torch.full(
            (prediction.size(0),1), batch_idx, dtype=prediction.dtype)
        prediction_dataloader_idx = torch.full(
            (prediction.size(0),1), dataloader_idx, dtype=prediction.dtype)


        prediction_np = prediction.numpy()
        
        # combine arrs
        out_arr = np.hstack((
            prediction_dataloader_idx.numpy(),
            prediction_batch_idx.numpy(),
            np.array(batch_indices).reshape(-1,1),
            prediction_np
        ))
        
        global_rank = pl_module.global_rank
        
        if f'dl-idx_b-idx_s-idx_pred_rank-{global_rank}' not in self.zarr_handle:
            self.zarr_handle.require_dataset(f'dl-idx_b-idx_s-idx_pred_rank-{global_rank}', 
                # prepend 3 integers (casted to float):
                # - dataloader_idx, 
                # - batch_idx, 
                # - batch_indices (sample index),
                # to prediction to avoid parallel write messing up with ordering
                shape=(0, 3 + self.emb_dim), 
                chunks=(10000, 3 + self.emb_dim), dtype='float')

        self.zarr_handle[f'dl-idx_b-idx_s-idx_pred_rank-{global_rank}'].append(out_arr)


        for ind, i in enumerate(batch_indices):
            self.json_handle.write(json.dumps({
                'embedding': prediction_np[ind].tolist(),
                'batch_idx': batch_idx, 
                'dataloader_idx': dataloader_idx, 
                'batch_indices': i
            }) + '\n')



class LLM_Inference_Base(pl.LightningModule):
    def __init__(self, llm_type: str, ckpt_path: str, config_path: str=None,
            output_dir: str=None, write_interval: str="batch", is_finetuned: bool = True, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        if self.hparams.is_finetuned:
            self.load_llm_ckpt(llm_type, ckpt_path, config_path)
        else:
            self.load_llm_pretrained(llm_type, ckpt_path, **kwargs)

    def load_llm_pretrained(self, llm_type, model_name, **kwargs):
        self.hparams.llm_type, self.hparams.ckpt_path = llm_type, model_name
        if llm_type.startswith('emb'):
            self.llm = LLM_Embed(model_name, **kwargs)
        elif llm_type.startswith('clm'):
            self.llm = LLM(model_name, **kwargs)
        elif llm_type.startswith('seqclf'):
            self.llm = LLM_SeqClassify(model_name, **kwargs)
        else:
            raise NotImplemented()

    def load_llm_ckpt(self, llm_type, ckpt_path, config_path):
        self.hparams.llm_type, self.hparams.ckpt_path, self.hparams.config_path = \
            llm_type, ckpt_path, config_path
        self.llm_config = yaml.safe_load(open(config_path, 'r'))

        if self.llm_config['trainer']['strategy'] is not None and \
            'deepspeed' in self.llm_config['trainer']['strategy']:
            tmp_path = os.path.join(ckpt_path, 'pytorch_model.bin')
            if not os.path.exists(tmp_path):
                logging.info(f"Creating {tmp_path} from deepspeed shards")
                convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, tmp_path)
            else:
                logging.info(f"Use already converted ckpt {tmp_path} from deepspeed shards")
            if llm_type.startswith('emb'):
                self.llm = LLM_Embed.load_from_checkpoint(tmp_path, strict=False)
            elif llm_type.startswith('clm'):
                self.llm = LLM.load_from_checkpoint(tmp_path, strict=False)
            elif llm_type.startswith('seqclf'):
                self.llm = LLM_SeqClassify.load_from_checkpoint(tmp_path, strict=False)
            else:
                raise NotImplemented()
        else:
            if llm_type.startswith('emb'):
                self.llm = LLM_Embed.load_from_checkpoint(ckpt_path, strict=False)
            elif llm_type.startswith('clm'):
                self.llm = LLM.load_from_checkpoint(ckpt_path, strict=False)
            elif llm_type.startswith('seqclf'):
                self.llm = LLM_SeqClassify.load_from_checkpoint(ckpt_path, strict=False)
            else:
                raise NotImplemented()
    
class LLM_Inference_Conditional_LM(LLM_Inference_Base):
    def __init__(self, llm_type: str, ckpt_path: str, config_path: str=None, output_dir: str=None,
                write_interval: str="batch", is_finetuned: bool=True, num_beams: int=3, 
                num_return_sequences: int=3, do_sample: bool=False, length_penalty: float=0, 
                max_new_tokens: int=50, allowed_gen_sequences: Union[List[str], str]=None
            ):
        buffer = []

        if allowed_gen_sequences is not None and isinstance(allowed_gen_sequences, str):
            with open(allowed_gen_sequences, 'r') as f:
                for l in f:
                    l = l.strip()
                    if len(l):
                        buffer.append(l)
        
        allowed_gen_sequences = buffer

        super().__init__(
            llm_type=llm_type,
            ckpt_path=ckpt_path,
            config_path=config_path,
            output_dir=output_dir,
            write_interval=write_interval,
            is_finetuned=is_finetuned,
            num_beams=num_beams, 
            num_return_sequences=num_return_sequences,
            do_sample=do_sample, 
            length_penalty=length_penalty, 
            max_new_tokens=max_new_tokens, 
            allowed_gen_sequences=allowed_gen_sequences
        )
        self.load_trie(allowed_gen_sequences)
    
    def load_trie(self, allowed_gen_sequences):
        self.hparams.allowed_gen_sequences = allowed_gen_sequences
        if allowed_gen_sequences is None or len(allowed_gen_sequences) == 0:
            self.trie = None
        else:
            self.trie = Trie([
                [self.llm.tokenizer.pad_token_id] + self.llm.tokenizer.encode(i) + \
                    [self.llm.tokenizer.eos_token_id] for i in allowed_gen_sequences
            ])
    
    def forward(self, input_ids, attention_mask):
        """Always return padded predictions with dim(1) == max_new_tokens"""
        if self.trie is None:
            prediction = self.llm.transformer.generate(
                input_ids = input_ids, 
                attention_mask = attention_mask,
                num_beams = self.hparams.num_beams, 
                num_return_sequences = self.hparams.num_return_sequences, 
                do_sample = self.hparams.do_sample, 
                length_penalty = self.hparams.length_penalty, 
                max_new_tokens = self.hparams.max_new_tokens - 1 # HACK: T5 adds pad token in the beginning
            )
        else:
            def constraint(batch_id, sent):
                return self.trie.get(sent.tolist())

            prediction = self.llm.transformer.generate(
                input_ids = input_ids, 
                attention_mask = attention_mask,
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

        return prediction_padded

    def predict_step(self, batch, batch_idx):
        return self.forward(batch['input_ids'], batch['attention_mask'])

class LLM_Inference_Embedding(LLM_Inference_Base):
    def __init__(self, llm_type: str, ckpt_path: str, config_path: str=None, output_dir: str=None, 
            write_interval: str="batch", is_finetuned: bool=True, is_input_embedding: bool=True, 
            hidden_states_type: str=None
        ):
        if is_finetuned:
            assert hidden_states_type is None
            super().__init__(
                llm_type=llm_type,
                ckpt_path=ckpt_path,
                config_path=config_path,
                output_dir=output_dir,
                write_interval=write_interval,
                is_finetuned=is_finetuned,
                is_input_embedding=is_input_embedding
            )
        else:
            assert hidden_states_type is not None
            super().__init__(
                llm_type=llm_type,
                ckpt_path=ckpt_path,
                config_path=config_path,
                output_dir=output_dir,
                write_interval=write_interval,
                is_finetuned=is_finetuned,
                is_input_embedding=is_input_embedding,
                hidden_states_type=hidden_states_type
            )
    
    def forward(self, input_ids, attention_mask):
        hidden_states = self.llm.get_hidden_states(
            input_ids = input_ids, 
            attention_mask = attention_mask
        )
        return hidden_states

    def predict_step(self, batch, batch_idx):
        if self.hparams.is_input_embedding:
            return self.forward(batch['input_ids'], batch['attention_mask'])
        else:
            return self.forward(batch['output_input_ids'], batch['output_attention_mask'])

#%%
class LLM_Inference_SeqClassify(LLM_Inference_Base):
    def __init__(self, llm_type: str, ckpt_path: str, config_path: str=None, output_dir: str=None, 
            write_interval: str="batch", is_finetuned: bool=True
        ):
        assert is_finetuned # doesn't make sense to have an unfinetuned seq clf llm
        super().__init__(
            llm_type=llm_type,
            ckpt_path=ckpt_path,
            config_path=config_path,
            output_dir=output_dir,
            write_interval=write_interval,
            is_finetuned=is_finetuned
        )
    
    def forward(self, input_ids, attention_mask):
        hidden_states = self.llm.get_hidden_states(
            input_ids = input_ids, 
            attention_mask = attention_mask
        )
        logits = self.llm.clf_head(hidden_states)
        return logits

    def predict_step(self, batch, batch_idx):
        return self.forward(batch['input_ids'], batch['attention_mask'])
        

#%%
@rank_zero_only
def clean_exisiting_inference_artifacts(output_dir, fname_prefix):
    logging.info(f"Cleaning existing inference artifacts with prefix {fname_prefix}")
    zarr_dir_path = pathlib.Path(os.path.join(output_dir, fname_prefix + '.zarr'))
    json_file_path = pathlib.Path(os.path.join(output_dir, fname_prefix + '.json'))

    try:
        shutil.rmtree(zarr_dir_path)
        logging.info(f"Successfully removed {zarr_dir_path}")
    except Exception as e:
        logging.info(f"Unable to remove {zarr_dir_path} due to {e}")
    try:
        os.remove(json_file_path)
        logging.info(f"Successfully removed {json_file_path}")
    except Exception as e:
        logging.info(f"Unable to remove {json_file_path} due to {e}")

@rank_zero_only
def end_of_inference_logging(output_dir, fname_prefix):
    logging.info("Saved inference results to {} and {}".format(
        os.path.join(output_dir, fname_prefix + '.json'),
        os.path.join(output_dir, fname_prefix + '.zarr')
    ))

# @rank_zero_only
# def log_cpu_memory_usage():
#     print('-----------')
#     import objgraph
#     objgraph.show_growth(limit=3)
#     print('-----------')

#%%
def cli_main():
    """LightningCLI handles CLI with no boilerplate:
    Please read https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli_advanced_3.html?highlight=class_path#models-with-multiple-submodules 

    1. Example command for conditional LM inference to infer

    python main_inference.py \
        --model LLM_Inference_Conditional_LM \
        --model.llm_type "clm" \
        --model.ckpt_path "models/product_title_multitask/version_1/epoch=0-step=349636.ckpt" \
        --model.config_path "models/product_title_multitask/version_1/config.yaml" \
        --model.output_dir="models/product_title_multitask/version_1" \
        --model.write_interval="batch" \
        --data LLMData \
        --data.data_source_yaml_path "datasets/product_title_taxonomy_classification/wish-tahoe-dedup-pseudo-test-simpleprompt-topdown.yaml" \
        --data.model_name "t5-base" \
        --data.batch_size=70 \
        --data.num_workers=8 \
        --trainer.logger=false \
        --trainer.enable_checkpointing=false \
        --trainer.accelerator gpu \
        --trainer.strategy ddp

    2. Example command for embedding LM inference to infer

    python main_inference.py \
        --model=LLM_Inference_Embedding \
        --model.llm_type="emb" \
        --model.ckpt_path="models/product_title_multitask/version_1/epoch=0-step=349636.ckpt" \
        --model.config_path="models/product_title_multitask/version_1/config.yaml" \
        --model.output_dir="models/product_title_multitask/version_1" \
        --model.write_interval="batch" \
        --model.is_input_embedding=true \
        --data=LLM_EmbedData \
        --data.data_source_yaml_path="datasets/product_title_embedding/wish-tahoe-dedup-pseudo-test.yaml" \
        --data.model_name="t5-base" \
        --data.batch_size=1500 \
        --data.num_workers=8 \
        --trainer.logger=false \
        --trainer.enable_checkpointing=false \
        --trainer.accelerator gpu \
        --trainer.strategy ddp

    3. Example command for un-finetuned embedding LM

    python main_inference.py \
        --model=LLM_Inference_Embedding \
        --model.llm_type="emb" \
        --model.ckpt_path="sentence-transformers/all-mpnet-base-v2" \
        --model.output_dir="models/product_title_embedding/unfinetuned" \
        --model.write_interval="batch" \
        --model.is_finetuned=false \
        --model.is_input_embedding=true \
        --model.hidden_states_type="encoder-mean" \
        --data=LLM_EmbedData \
        --data.data_source_yaml_path="datasets/product_title_embedding/wish-mturk-test.yaml" \
        --data.model_name="sentence-transformers/all-mpnet-base-v2" \
        --trainer.logger=false \
        --trainer.enable_checkpointing=false \
        --trainer.accelerator gpu \
        --trainer.strategy ddp

    4. Example command for finetuned seqclf LM

    python main_inference.py \
        --model=LLM_Inference_SeqClassify \
        --model.llm_type="seqclf" \
        --model.ckpt_path="models/product_title_multitask/version_1/epoch=0-step=349636.ckpt" \
        --model.config_path="models/product_title_multitask/version_1/config.yaml" \
        --model.output_dir="models/product_title_multitask/version_1" \
        --model.write_interval="batch" \
        --model.is_finetuned=true \
        --data=LLM_SeqClassifyData \
        --data.data_source_yaml_path="datasets/product_title_seqclf/wish-offshore-validated-test.yaml" \
        --data.model_name="t5-base" \
        --data.label_map_file="datasets/taxonomy/wish_v1.2.1_newtax_allpaths.txt" \
        --trainer.logger=false \
        --trainer.enable_checkpointing=false \
        --trainer.accelerator gpu \
        --trainer.strategy ddp

    5. Example command for finetuned seqclf LM on inputonly data for prediction only 
        (Use LLM_SeqClassifyInputOnlyData for pseudolabel generation type of inference)
        (LLM_SeqClassifyData would instead waste time mapping labels, useless for pseudolabel-generation type of inference)
    
    python main_inference.py \
        --model=LLM_Inference_SeqClassify \
        --model.llm_type="seqclf" \
        --model.ckpt_path="models/product_title_multitask/version_1/epoch=0-step=349636.ckpt" \
        --model.config_path="models/product_title_multitask/version_1/config.yaml" \
        --model.output_dir="models/product_title_multitask/version_1" \
        --model.write_interval="batch" \
        --model.is_finetuned=true \
        --data=LLM_SeqClassifyInputOnlyData \
        --data.data_source_yaml_path="datasets/product_title_seqclf/wish-tahoe-dedup-train-predict-inputonly.yaml" \
        --data.model_name="t5-base" \
        --data.label_map_file="datasets/taxonomy/wish_v1.2.1_newtax_allpaths.txt" \
        --data.batch_size=1500 \
        --data.num_workers=8 \
        --trainer.logger=false \
        --trainer.enable_checkpointing=false \
        --trainer.accelerator gpu \
        --trainer.strategy ddp
    """
    cli = LightningCLI(run=False, save_config_callback=None)
    transformer_config, _ = get_transformer(cli.config.data.init_args.model_name, return_model=False)
    if cli.model.hparams.is_finetuned:
        fname_prefix = (cli.model.hparams.ckpt_path.split('/')[-1].replace('.ckpt', '') + '--' + 
                cli.datamodule.hparams.data_source_yaml_path.split('/')[-1].replace('.yaml', ''))  
    else:
        fname_prefix = (cli.model.hparams.ckpt_path.replace('/', '-') + '--' + 
                cli.datamodule.hparams.data_source_yaml_path.split('/')[-1].replace('.yaml', '')) 
    if cli.config.model.init_args.llm_type.startswith('emb'):
        if cli.config.model.init_args.is_input_embedding:
            fname_prefix = fname_prefix + '--inputemb'
        else:
            fname_prefix = fname_prefix + '--outputemb'
    fname_prefix = cli.config.model.init_args.llm_type + '-' + fname_prefix
    clean_exisiting_inference_artifacts(cli.config.model.init_args.output_dir, fname_prefix)

    if cli.config.model.init_args.llm_type.startswith('clm'):
        write_callback = PredictionWriter_Conditional_LM(
            output_dir=cli.config.model.init_args.output_dir,
            write_interval=cli.config.model.init_args.write_interval,
            fname_prefix=fname_prefix, 
            max_new_tokens=cli.config.model.init_args.max_new_tokens,
            num_return_sequences=cli.config.model.init_args.num_return_sequences
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