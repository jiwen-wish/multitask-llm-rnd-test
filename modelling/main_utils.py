#%%
from typing import Dict, List
from collections import OrderedDict
from copy import deepcopy
import logging
import yaml
import gzip
import os
import glob
import pathlib
import json
import hashlib
import dvc.api
import multiprocessing
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
from datasets import load_dataset, Value, Features, DatasetDict, Dataset
from torch import utils
import random
from thefuzz import process as fuzz_process
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModel,
    AutoTokenizer
)
import time
# asyncssh is too verbose
logging.getLogger("asyncssh").disabled = True

def get_transformer(model_name, return_model=True, **kwargs):
    config, unused_kwargs = AutoConfig.from_pretrained(model_name, return_unused_kwargs=True, **kwargs)
    logging.info(f"Unused kwargs when getting {model_name}: {unused_kwargs}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if 'gpt2' in model_name:
        tokenizer.pad_token = tokenizer.eos_token
    if not return_model:
        return config, tokenizer
    else:
        if config.is_encoder_decoder:
            model = AutoModelForSeq2SeqLM.from_config(config)
        else:
            model = AutoModel.from_config(config)
        return config, model, tokenizer

def sortOD(od):
    res = OrderedDict()
    for k, v in sorted(od.items()):
        if isinstance(v, dict):
            res[k] = sortOD(v)
        else:
            res[k] = deepcopy(v)
    return res

class LLMData(pl.LightningDataModule):
    """Data for Conditional Language Modelling"""
    def __init__(self, data_source_yaml_path: str, model_name: str = 't5-base', 
            raw_cache_dir: str = os.environ["GENERAL_CACHE"] if "GENERAL_CACHE" in os.environ else None, 
            batch_size: int = 16, 
            overwrite_cache: bool = False, 
            max_length: int = 250,
            predict_on_test: bool = True,
            num_workers: int = multiprocessing.cpu_count(),
            max_length_out: int = 100,
            **kwargs):
        assert raw_cache_dir is not None, "cahce_dir cannot be None"
        super().__init__()
        self.save_hyperparameters()
        self.transformer_config, self.tokenizer = get_transformer(
            self.hparams.model_name, return_model=False, **kwargs)
        data_source_dict = yaml.safe_load(open(self.hparams.data_source_yaml_path, 'r'))
        data_source_dict['is_encoder_decoder'] = self.transformer_config.is_encoder_decoder is True
        self.hparams.data_source = sortOD(data_source_dict)
        self.hparams.data_hash = hashlib.md5(
            json.dumps(self.hparams.data_source).encode('utf-8')
        ).hexdigest()
        self.hparams.raw_cache_dir_folder = os.path.join(self.hparams.raw_cache_dir, 
            self.hparams.data_hash)
        pathlib.Path(self.hparams.raw_cache_dir_folder).mkdir(parents=True, exist_ok=True)
    
    def transform_write_datum(self, dat, file_dict, fout):
        in_, out_ = dat['text'].split(' -> ')
        # TODO: formalize this
        if self.hparams.data_source['preprocess']['transform'] == 'bidirectional':
            list_out_ = out_[1:-1].split('][')
            fout.write(
                (json.dumps({"text_input": "Top-down " + file_dict['task_prefix'] + in_, 
                "text_output": ' > '.join(list_out_)}) + '\n').encode('utf-8')
            )
            fout.write(
                (json.dumps({"text_input": "Bottom-up " + file_dict['task_prefix'] + in_, 
                "text_output": ' < '.join(list_out_[::-1])}) + '\n').encode('utf-8')
            )
        elif self.hparams.data_source['preprocess']['transform'] == 'top-down':
            list_out_ = out_[1:-1].split('][')
            fout.write(
                (json.dumps({"text_input": "Top-down " + file_dict['task_prefix'] + in_, 
                "text_output": ' > '.join(list_out_)}) + '\n').encode('utf-8')
            )
        elif self.hparams.data_source['preprocess']['transform'] == 'bottom-up':
            list_out_ = out_[1:-1].split('][')
            fout.write(
                (json.dumps({"text_input": "Bottom-up " + file_dict['task_prefix'] + in_, 
                "text_output": ' < '.join(list_out_[::-1])}) + '\n').encode('utf-8')
            )
        elif self.hparams.data_source['preprocess']['transform'] == 'nothing':
            fout.write(
                (json.dumps({"text_input": file_dict['task_prefix'] + in_, 
                "text_output": out_}) + '\n').encode('utf-8')
            )
        else:
            raise NotImplemented()

    def prepare_data(self):
        # important: raw json files needs to contain "text" key
        existing_files = glob.glob(self.hparams.raw_cache_dir_folder + '/*.json.gz')
        existing_files_short = [i.split('/')[-1] for i in existing_files]
        if len(existing_files) != len([i for i in self.hparams.data_source if i in ['train', 'val', 'test']]) or \
                self.hparams.overwrite_cache:
            logging.info(f"loading {self.hparams.data_source_yaml_path} from dvc")
            for stage in self.hparams.data_source:
                if stage in ['train', 'val', 'test']:
                    if f'{stage}.json.gz' in existing_files_short and not self.hparams.overwrite_cache:
                        logging.info(f"Found existing {stage}.json.gz in {self.hparams.raw_cache_dir_folder}, Skip")
                    else:
                        logging.info(f"Write {stage}.json.gz to {self.hparams.raw_cache_dir_folder}")
                        for try_num in range(5):
                            try:
                                outfile_path = os.path.join(self.hparams.raw_cache_dir_folder, 
                                    f'{stage}.json.gz')
                                with gzip.open(outfile_path, 'w') as fout:
                                    for file_dict in self.hparams.data_source[stage]:
                                        logging.info(f"Download dvc {file_dict['path']} in {file_dict['repo']} from {self.hparams.data_source_yaml_path} now...")
                                        with dvc.api.open(
                                            path=file_dict['path'],
                                            repo=file_dict['repo'],
                                            rev=file_dict['rev']
                                        ) as f:
                                            if not self.transformer_config.is_encoder_decoder:
                                                logging.warning("encoder-only / decoder-only model development stopped, "
                                                    "please only use encoder-decoder models unless using embedding model")
                                            for l in tqdm(f):
                                                dat = json.loads(l)
                                                # important: text needs to contain exactly one " -> " for encoder-decoder model
                                                try:
                                                    self.transform_write_datum(dat, file_dict, fout)
                                                except Exception as e:
                                                    logging.warning('Skip ' + dat['text'] + f" due to {e}")  
                                logging.info(f"Successfully write {stage}.json.gz to {self.hparams.raw_cache_dir_folder}")
                                break
                            except Exception as e:
                                logging.warning(f'Try {try_num} failed with {e}, rerun Write {stage}.json.gz to {self.hparams.raw_cache_dir_folder}')
                                logging.warning(f"Remove failed {outfile_path}")
                                os.remove(outfile_path)
                                time.sleep(10)

        else:
            logging.info(f"Use cache stored in {self.hparams.raw_cache_dir_folder} for {self.hparams.data_source_yaml_path}")

    def get_hf_dataset(self):
        logging.info(f"Convert data in {self.hparams.raw_cache_dir_folder} to Huggingface dataset")
        try:
            ds = load_dataset('json', data_files={
                stage: os.path.join(self.hparams.raw_cache_dir_folder, f'{stage}.json.gz') for stage \
                    in self.hparams.data_source if stage in ['train', 'val', 'test']
            }, features=Features(text_input = Value(dtype="string", id=None), text_output = Value(dtype="string", id=None)))

            if 0 in [len(ds[i]) for i in ds]:
                logging.info("Native datasets load failed, use pandas to manually create huggingface dataset")
                ds = DatasetDict({
                    stage: Dataset.from_pandas(
                        pd.read_json(os.path.join(self.hparams.raw_cache_dir_folder, f'{stage}.json.gz'), lines=True)) for stage \
                            in self.hparams.data_source if stage in ['train', 'val', 'test']
                })
        except:
            logging.info("Native datasets load failed, use pandas to manually create huggingface dataset")
            ds = DatasetDict({
                stage: Dataset.from_pandas(
                    pd.read_json(os.path.join(self.hparams.raw_cache_dir_folder, f'{stage}.json.gz'), lines=True)) for stage \
                        in self.hparams.data_source if stage in ['train', 'val', 'test']
            })
        logging.info(f"hf dataset: {ds}")
        return ds

    def setup(self, stage: str):
        ds = self.get_hf_dataset()
        if self.transformer_config.is_encoder_decoder:
            def seq2seq_encode(examples):
                input_ = self.tokenizer(examples['text_input'], return_tensors='pt', 
                    padding="max_length", truncation=True, 
                    max_length=self.hparams.max_length)
                output_ = self.tokenizer(examples['text_output'], return_tensors='pt', 
                    padding="max_length", truncation=True, 
                    max_length=self.hparams.max_length_out)
                labels = output_.input_ids
                labels[labels == self.tokenizer.pad_token_id] = -100
                input_['labels'] = labels
                return input_

            ds.set_transform(seq2seq_encode)
        else:
            raise Exception("encoder-only / decoder-only model development stopped, "
                "please only use encoder-decoder models unless using embedding model")

        self.ds = ds
    
    def train_dataloader(self):
        if 'train' in self.ds:
            return utils.data.DataLoader(
                    self.ds['train'], batch_size=self.hparams.batch_size, shuffle=True,
                    num_workers=self.hparams.num_workers, pin_memory=True
                )

    def val_dataloader(self):
        if 'val' in self.ds:
            return utils.data.DataLoader(
                self.ds['val'], batch_size=self.hparams.batch_size, shuffle=False,
                num_workers=self.hparams.num_workers, pin_memory=True
            )

    def test_dataloader(self):
        if 'test' in self.ds:
            return utils.data.DataLoader(
                self.ds['test'], batch_size=self.hparams.batch_size, shuffle=False,
                num_workers=self.hparams.num_workers, pin_memory=True
            )

    def predict_dataloader(self):
        if self.hparams.predict_on_test and 'test' in self.ds:
            return utils.data.DataLoader(
                self.ds['test'], batch_size=self.hparams.batch_size, shuffle=False,
                num_workers=self.hparams.num_workers, pin_memory=True
            )

#%%
class LLM_DenoiseData(LLMData):
    """Data for Denoising Language Modelling"""
    def __init__(self, data_source_yaml_path: str, model_name: str = 't5-base', 
            raw_cache_dir: str = os.environ["GENERAL_CACHE"] if "GENERAL_CACHE" in os.environ else None, 
            batch_size: int = 32, 
            overwrite_cache: bool = False, 
            max_length: int = 50,
            predict_on_test: bool = True,
            num_workers: int = multiprocessing.cpu_count(),
            max_length_out: int = 50,
            mask_prob: float = 0.3,
            use_ul2: bool = False, # https://arxiv.org/pdf/2205.05131.pdf and https://arxiv.org/pdf/2210.11399.pdf
            **kwargs):
            super().__init__(
                data_source_yaml_path=data_source_yaml_path,
                model_name=model_name,
                raw_cache_dir=raw_cache_dir,
                batch_size=batch_size,
                overwrite_cache=overwrite_cache,
                max_length=max_length,
                predict_on_test=predict_on_test,
                num_workers=num_workers,
                max_length_out=max_length_out,
                mask_prob=mask_prob,
                use_ul2=use_ul2
            )
            self.local_additional_special_tokens_list = deepcopy(self.tokenizer.additional_special_tokens)

    @staticmethod
    def mask_input(input_text, base_mask_prob, use_ul2, additional_special_tokens_list, 
            use_causal=False, ul2_prefix='', use_task_prefix=True, return_num_special_tokens_inserted=False):
        """T5-style denoising
        input_text:    Denoise text: The cute dog walks in the park
        -->
        masked_input:  Denoise text: The <extra_id_0> walks in <extra_id_1> park
        target_output: <extra_id_0> cute dog <extra_id_1> the <extra_id_2>

        base_mask_prob - probaility of a word (not token) being masked for regular mask corruption task
        use_ul2 - use mixture of denoisers as in https://arxiv.org/pdf/2205.05131.pdf and https://arxiv.org/pdf/2210.11399.pdf, 
        # albeit significantly simplified
        """
        if use_ul2:
            # route to randomly selected denoiser
            rnd = np.random.random()
            if rnd < .2:
                # sequential (causal) denoiser
                return LLM_DenoiseData.mask_input(input_text, base_mask_prob, not use_ul2, 
                    additional_special_tokens_list, use_causal=True, ul2_prefix='S2S ', use_task_prefix=use_task_prefix,
                    return_num_special_tokens_inserted=return_num_special_tokens_inserted
                )
            elif rnd >= .2 and rnd < .4:
                # extreme denoiser
                return LLM_DenoiseData.mask_input(input_text, np.random.uniform(low=base_mask_prob, high=.9), not use_ul2, 
                    additional_special_tokens_list, use_causal=False, ul2_prefix='NLG ', use_task_prefix=use_task_prefix,
                    return_num_special_tokens_inserted=return_num_special_tokens_inserted
                )
            else:
                # regular denoiser
                ul2_prefix='NLU '
        if use_task_prefix:
            task_prompt = input_text.split(": ")[0] + ": "
            input_text = ": ".join(input_text.split(": ")[1:])
        masked_input, target_output = [], [] 
        space_delim = True
        if " " in input_text:
            words = input_text.split(" ")
        else:
            words = list(input_text)
            space_delim = False
        # return None if input_text is bad or no special tokens left
        if len(words) <= 1 or '<extra_id_' in input_text or len(additional_special_tokens_list) == 0:
            return None 
        else:
            # mask at least one word, at max T5-handlable words
            if use_causal:
                mask_word_indices = set(range(
                    len(words) - np.random.randint(low=1, high=len(words)), len(words)
                ))
            else:
                mask_word_indices = set(random.sample(
                    range(len(words)), 
                    min(
                        max(1, int(base_mask_prob * len(words))), 
                        len(additional_special_tokens_list) - 1
                    )
                ))
            extra_id_c = 0
            for ind, w in enumerate(words):
                cur_mask_token = additional_special_tokens_list[extra_id_c]
                # if current word masked
                if ind in mask_word_indices:
                    # if previous word also masked, only modify target_output 
                    if ind -1 in mask_word_indices:
                        target_output.append(w)
                    # if previous word not masked, modify: masked_input, target_output and cur_mask_token
                    else:
                        # add special token to target_output
                        target_output.append(cur_mask_token)
                        # add word to target_output
                        target_output.append(w)
                        # add special token to masked_input
                        masked_input.append(cur_mask_token)
                        # increment cur_mask_token counter
                        extra_id_c += 1
                # if current word not masked, only modify masked_input
                else:
                    masked_input.append(w)
            # add ending special token
            target_output.append(additional_special_tokens_list[extra_id_c])
            
            delim = ""
            task_prompt_ = ""
            ul2_prefix_ = ""

            if space_delim:
                delim = " "
            if use_task_prefix:
                ul2_prefix_ = ul2_prefix
                task_prompt_ = task_prompt

            if return_num_special_tokens_inserted:
                return ul2_prefix_ + task_prompt_ + delim.join(masked_input), delim.join(target_output), extra_id_c
            else:
                return ul2_prefix_ + task_prompt_ + delim.join(masked_input), delim.join(target_output)
            
            # if return_num_special_tokens_inserted:
            #     if use_task_prefix:
            #         if space_delim:
            #             return ul2_prefix + task_prompt + " ".join(masked_input), " ".join(target_output), extra_id_c
            #         else:
            #             return ul2_prefix + task_prompt + "".join(masked_input), "".join(target_output), extra_id_c
            #     else:
            #         return " ".join(masked_input), " ".join(target_output), extra_id_c
            # else:
            #     if use_task_prefix:
            #         if space_delim:
            #             return ul2_prefix + task_prompt + " ".join(masked_input), " ".join(target_output)
            #         else:
            #             return ul2_prefix + task_prompt + "".join(masked_input), "".join(target_output)
            #     else:
            #         return " ".join(masked_input), " ".join(target_output)

    def transform_write_datum(self, dat, file_dict, fout):
        in_, _ = dat['text'].split(' -> ')
        fout.write(
            (json.dumps({"text_input": file_dict['task_prefix'] + in_}) + '\n').encode('utf-8')
        )
    
    def get_hf_dataset(self):
        logging.info(f"Convert data in {self.hparams.raw_cache_dir_folder} to Huggingface dataset")
        try:
            ds = load_dataset('json', data_files={
                stage: os.path.join(self.hparams.raw_cache_dir_folder, f'{stage}.json.gz') for stage \
                    in self.hparams.data_source if stage in ['train', 'val', 'test']
            }, features=Features(text_input = Value(dtype="string", id=None)))

            if 0 in [len(ds[i]) for i in ds]:
                logging.info("Native datasets load failed, use pandas to manually create huggingface dataset")
                ds = DatasetDict({
                    stage: Dataset.from_pandas(
                        pd.read_json(os.path.join(self.hparams.raw_cache_dir_folder, f'{stage}.json.gz'), lines=True)) for stage \
                            in self.hparams.data_source if stage in ['train', 'val', 'test']
                })
        except:
            logging.info("Native datasets load failed, use pandas to manually create huggingface dataset")
            ds = DatasetDict({
                stage: Dataset.from_pandas(
                    pd.read_json(os.path.join(self.hparams.raw_cache_dir_folder, f'{stage}.json.gz'), lines=True)) for stage \
                        in self.hparams.data_source if stage in ['train', 'val', 'test']
            })
        logging.info(f"hf dataset: {ds}")
        return ds

    def setup(self, stage: str):
        ds = self.get_hf_dataset()
        if self.transformer_config.is_encoder_decoder:
            def seq2seq_denoise_encode(examples):
                mask_res = [LLM_DenoiseData.mask_input(i, self.hparams.mask_prob, self.hparams.use_ul2, 
                    self.local_additional_special_tokens_list) for i in examples['text_input']]
                mask_res = [i for i in mask_res if i is not None]
                if len(mask_res):
                    masked_input, target_output = zip(*mask_res)
                    input_ = self.tokenizer(list(masked_input), return_tensors='pt', 
                        padding="max_length", truncation=True, 
                        max_length=self.hparams.max_length)
                    output_ = self.tokenizer(list(target_output), return_tensors='pt', 
                        padding="max_length", truncation=True, 
                        max_length=self.hparams.max_length_out)
                    labels = output_.input_ids
                    labels[labels == self.tokenizer.pad_token_id] = -100
                    input_['labels'] = labels
                    return input_
                else:
                    # return dummy data if no texts in sample can be denoised
                    logging.warning(f"No examples in {examples} can be denoised")
                    return {
                        'input_ids': torch.full(
                            size=(1, self.hparams.max_length_out),
                            fill_value= self.tokenizer.pad_token_id,
                            dtype=torch.long
                        ),
                        'attention_mask': torch.full(
                            size=(1, self.hparams.max_length_out),
                            fill_value= 0,
                            dtype=torch.long
                        ),
                        'labels': torch.full(
                            size=(1, self.hparams.max_length_out),
                            fill_value= -100,
                            dtype=torch.long
                        )
                    }

            ds.set_transform(seq2seq_denoise_encode)
        else:
            raise Exception("encoder-only / decoder-only model development stopped, "
                "please only use encoder-decoder models unless using embedding model")

        self.ds = ds

#%%
class LLM_EmbedData(LLMData):
    """Data for CLIP-style In-Batch Contrastive Text Embedding"""
    def __init__(self, data_source_yaml_path: str, model_name: str = 't5-base', 
            raw_cache_dir: str = os.environ["GENERAL_CACHE"] if "GENERAL_CACHE" in os.environ else None, 
            batch_size: int = 32, 
            overwrite_cache: bool = False, 
            max_length: int = 50,
            predict_on_test: bool = True,
            num_workers: int = multiprocessing.cpu_count(),
            max_length_out: int = 50,
            **kwargs):
            super().__init__(
                data_source_yaml_path=data_source_yaml_path,
                model_name=model_name,
                raw_cache_dir=raw_cache_dir,
                batch_size=batch_size,
                overwrite_cache=overwrite_cache,
                max_length=max_length,
                predict_on_test=predict_on_test,
                num_workers=num_workers,
                max_length_out=max_length_out,
            )

    def transform_write_datum(self, dat, file_dict, fout):
        in_, out_ = dat['text'].split(' -> ')
        list_out_ = out_[1:-1].split('][')
        fout.write(
            (json.dumps({"text_input": file_dict['task_prefix_input'] + in_, 
                "text_output": file_dict['task_prefix_output'] + ' > '.join(list_out_)}) + '\n').encode('utf-8')
        )
    
    def setup(self, stage: str):
        ds = self.get_hf_dataset()
        if not self.transformer_config.is_encoder_decoder:
            logging.warning("encoder-only / decoder-only model development stopped, "
                "please only use encoder-decoder models unless using embedding model")
        def seq2seq_encode(examples):
            input_ = self.tokenizer(examples['text_input'], return_tensors='pt', 
                padding="max_length", truncation=True, 
                max_length=self.hparams.max_length)
            output_ = self.tokenizer(examples['text_output'], return_tensors='pt', 
                padding="max_length", truncation=True, 
                max_length=self.hparams.max_length_out)
            for i in output_:
                input_[f'output_{i}'] = output_[i]
            return input_

        ds.set_transform(seq2seq_encode)
        self.ds = ds
    
    def train_dataloader(self):
        if 'train' in self.ds:
            return utils.data.DataLoader(
                self.ds['train'], batch_size=self.hparams.batch_size, shuffle=True,
                num_workers=self.hparams.num_workers, pin_memory=True
            )

    def val_dataloader(self):
        if 'val' in self.ds:
            return utils.data.DataLoader(
                self.ds['val'], batch_size=self.hparams.batch_size, shuffle=True,
                num_workers=self.hparams.num_workers, pin_memory=True
            )

    def test_dataloader(self):
        if 'test' in self.ds:
            return utils.data.DataLoader(
                self.ds['test'], batch_size=self.hparams.batch_size, shuffle=True,
                num_workers=self.hparams.num_workers, pin_memory=True
            )
    
    def predict_dataloader(self):
        if self.hparams.predict_on_test and 'test' in self.ds:
            return utils.data.DataLoader(
                self.ds['test'], batch_size=self.hparams.batch_size, shuffle=False,
                num_workers=self.hparams.num_workers, pin_memory=True
            )

#%%
class LLM_EmbedManualData(LLM_EmbedData):
    """Data for manual contrastive learning (not in-batch negative samples)"""
    def __init__(self, data_source_yaml_path: str, model_name: str = 't5-base', 
            raw_cache_dir: str = os.environ["GENERAL_CACHE"] if "GENERAL_CACHE" in os.environ else None, 
            batch_size: int = 32, 
            overwrite_cache: bool = False, 
            max_length: int = 50,
            predict_on_test: bool = True,
            num_workers: int = multiprocessing.cpu_count(),
            max_length_out: int = 50,
            **kwargs):
            super().__init__(
                data_source_yaml_path=data_source_yaml_path,
                model_name=model_name,
                raw_cache_dir=raw_cache_dir,
                batch_size=batch_size,
                overwrite_cache=overwrite_cache,
                max_length=max_length,
                predict_on_test=predict_on_test,
                num_workers=num_workers,
                max_length_out=max_length_out,
            )
    
    def transform_write_datum(self, dat, file_dict, fout):
        in_, out_ = dat['text_input'], dat['text_output']
        labels_ = dat['labels']
        fout.write(
            (json.dumps({"text_input": file_dict['task_prefix_input'] + in_, 
                "text_output": file_dict['task_prefix_output'] + out_, "labels": labels_}) + '\n').encode('utf-8')
        )
    
    def get_hf_dataset(self):
        logging.info(f"Convert data in {self.hparams.raw_cache_dir_folder} to Huggingface dataset")
        try:
            ds = load_dataset('json', data_files={
                stage: os.path.join(self.hparams.raw_cache_dir_folder, f'{stage}.json.gz') for stage \
                    in self.hparams.data_source if stage in ['train', 'val', 'test']
            }, features=Features(text_input = Value(dtype="string", id=None), text_output = Value(dtype="string", id=None), 
                labels = Value(dtype="float", id=None)))

            if 0 in [len(ds[i]) for i in ds]:
                logging.info("Native datasets load failed, use pandas to manually create huggingface dataset")
                ds = DatasetDict({
                    stage: Dataset.from_pandas(
                        pd.read_json(os.path.join(self.hparams.raw_cache_dir_folder, f'{stage}.json.gz'), lines=True)) for stage \
                            in self.hparams.data_source if stage in ['train', 'val', 'test']
                })
        except:
            logging.info("Native datasets load failed, use pandas to manually create huggingface dataset")
            ds = DatasetDict({
                stage: Dataset.from_pandas(
                    pd.read_json(os.path.join(self.hparams.raw_cache_dir_folder, f'{stage}.json.gz'), lines=True)) for stage \
                        in self.hparams.data_source if stage in ['train', 'val', 'test']
            })
        logging.info(f"hf dataset: {ds}")
        return ds

    def setup(self, stage: str):
        ds = self.get_hf_dataset()
        if not self.transformer_config.is_encoder_decoder:
            logging.warning("encoder-only / decoder-only model development stopped, "
                "please only use encoder-decoder models unless using embedding model")
        def seq2seq_encode(examples):
            input_ = self.tokenizer(examples['text_input'], return_tensors='pt', 
                padding="max_length", truncation=True, 
                max_length=self.hparams.max_length)
            output_ = self.tokenizer(examples['text_output'], return_tensors='pt', 
                padding="max_length", truncation=True, 
                max_length=self.hparams.max_length_out)
            labels_ = torch.FloatTensor(examples['labels'])
            for i in output_:
                input_[f'output_{i}'] = output_[i]
            input_['labels'] = labels_
            return input_

        ds.set_transform(seq2seq_encode)
        self.ds = ds
    
    def train_dataloader(self):
        if 'train' in self.ds:
            return utils.data.DataLoader(
                self.ds['train'], batch_size=self.hparams.batch_size, shuffle=True,
                num_workers=self.hparams.num_workers, pin_memory=True
            )

    def val_dataloader(self):
        if 'val' in self.ds:
            return utils.data.DataLoader(
                self.ds['val'], batch_size=self.hparams.batch_size, shuffle=False,
                num_workers=self.hparams.num_workers, pin_memory=True
            )

    def test_dataloader(self):
        if 'test' in self.ds:
            return utils.data.DataLoader(
                self.ds['test'], batch_size=self.hparams.batch_size, shuffle=False,
                num_workers=self.hparams.num_workers, pin_memory=True
            )
    
    def predict_dataloader(self):
        if self.hparams.predict_on_test and 'test' in self.ds:
            return utils.data.DataLoader(
                self.ds['test'], batch_size=self.hparams.batch_size, shuffle=False,
                num_workers=self.hparams.num_workers, pin_memory=True
            )

#%%
class LLM_SeqClassifyData(LLM_EmbedData):
    """Data for Sequence Classification"""
    def __init__(self, data_source_yaml_path: str, model_name: str = 't5-base', 
            raw_cache_dir: str = os.environ["GENERAL_CACHE"] if "GENERAL_CACHE" in os.environ else None, 
            batch_size: int = 32, 
            overwrite_cache: bool = False, 
            max_length: int = 50,
            predict_on_test: bool = True,
            num_workers: int = multiprocessing.cpu_count(),
            label_map_file: str = None,
            **kwargs):
        assert label_map_file is not None
        super().__init__(
            data_source_yaml_path=data_source_yaml_path,
            model_name=model_name,
            raw_cache_dir=raw_cache_dir,
            batch_size=batch_size,
            overwrite_cache=overwrite_cache,
            max_length=max_length,
            predict_on_test=predict_on_test,
            num_workers=num_workers,
            label_map_file=label_map_file
        )
        self.save_hyperparameters()
        self.label_map = {}
        with open(self.hparams.label_map_file, 'r') as f:
            for l in f:
                l = l.replace('\n', '').strip()
                if len(l):
                    self.label_map[l] = len(self.label_map)
        self.label_list = sorted([i for i in self.label_map.items()], key=lambda x: x[1])
        self.label_list = [i[0] for i in self.label_list]
        self.match_label_map = {}

    def text2label(self, text):
        """Convert plain text label to numerical label"""
        label = [0] * len(self.label_map)
        textlist = text.split(" > ")
        l_s = textlist[0]
        label[self.label_map[l_s]] = 1
        for l in textlist[1:]:
            l_s = l_s + " > " + l
            label[self.label_map[l_s]] = 1
        return label

    def transform_write_datum(self, dat, file_dict, fout):
        in_, out_ = dat['text'].split(' -> ')
        list_out_ = out_[1:-1].split('][')
        out_txt = ' > '.join(list_out_)
        
        if out_txt in self.label_map:
            fout.write(
                (json.dumps({"text_input": file_dict['task_prefix_input'] + in_, 
                    "text_output": out_txt}) + '\n').encode('utf-8')
            )
        elif out_txt in self.match_label_map:
            fout.write(
                (json.dumps({"text_input": file_dict['task_prefix_input'] + in_, 
                    "text_output": self.match_label_map[out_txt]}) + '\n').encode('utf-8')
            )
        else:
            out_txt_match = fuzz_process.extractOne(out_txt, self.label_list)
            logging.warning(f"{out_txt} not in label_map, matched to {out_txt_match}")
            self.match_label_map[out_txt] = out_txt_match[0]
            logging.warning(f"match_label_map grow to {len(self.match_label_map)}")
            fout.write(
                (json.dumps({"text_input": file_dict['task_prefix_input'] + in_, 
                    "text_output": out_txt_match[0]}) + '\n').encode('utf-8')
            )

    def setup(self, stage: str):
        ds = self.get_hf_dataset()
        if not self.transformer_config.is_encoder_decoder:
            logging.warning("encoder-only / decoder-only model development stopped, "
                "please only use encoder-decoder models unless using embedding model")
        def seq2seq_encode(examples):
            input_ = self.tokenizer(examples['text_input'], return_tensors='pt', 
                padding="max_length", truncation=True, 
                max_length=self.hparams.max_length)
            output_ = [self.text2label(i) for i in examples['text_output']]
            input_['labels'] = torch.FloatTensor(output_)
            return input_

        ds.set_transform(seq2seq_encode)
        self.ds = ds
    
    def train_dataloader(self):
        if 'train' in self.ds:
            return utils.data.DataLoader(
                self.ds['train'], batch_size=self.hparams.batch_size, shuffle=True,
                num_workers=self.hparams.num_workers, pin_memory=True
            )

    def val_dataloader(self):
        if 'val' in self.ds:
            return utils.data.DataLoader(
                self.ds['val'], batch_size=self.hparams.batch_size, shuffle=False,
                num_workers=self.hparams.num_workers, pin_memory=True
            )

    def test_dataloader(self):
        if 'test' in self.ds:
            return utils.data.DataLoader(
                self.ds['test'], batch_size=self.hparams.batch_size, shuffle=False,
                num_workers=self.hparams.num_workers, pin_memory=True
            )
    
    def predict_dataloader(self):
        if self.hparams.predict_on_test and 'test' in self.ds:
            return utils.data.DataLoader(
                self.ds['test'], batch_size=self.hparams.batch_size, shuffle=False,
                num_workers=self.hparams.num_workers, pin_memory=True
            )

class LLM_SeqClassifyInputOnlyData(LLM_EmbedData):
    def __init__(self, data_source_yaml_path: str, model_name: str = 't5-base', 
            raw_cache_dir: str = os.environ["GENERAL_CACHE"] if "GENERAL_CACHE" in os.environ else None, 
            batch_size: int = 32, 
            overwrite_cache: bool = False, 
            max_length: int = 50,
            num_workers: int = multiprocessing.cpu_count(),
            label_map_file: str = None,
            **kwargs):
        super().__init__(
            data_source_yaml_path=data_source_yaml_path,
            model_name=model_name,
            raw_cache_dir=raw_cache_dir,
            batch_size=batch_size,
            overwrite_cache=overwrite_cache,
            max_length=max_length,
            num_workers=num_workers
        )
        self.save_hyperparameters()
        logging.info(f"label_map_file {label_map_file} is unused inside LLM_SeqClassifyInputOnlyData")
        assert 'predict' in self.hparams.data_source, f"LLM_SeqClassifyInputOnlyData must have predict in {data_source_yaml_path} for inference purpose"
    
    def prepare_data(self):
        # important: raw json files needs to contain "text" key
        existing_files = glob.glob(self.hparams.raw_cache_dir_folder + '/*.json.gz')
        existing_files_short = [i.split('/')[-1] for i in existing_files]
        if len(existing_files) != len([i for i in self.hparams.data_source if i in ['predict']]) or \
                self.hparams.overwrite_cache:
            logging.info(f"loading {self.hparams.data_source_yaml_path} from dvc")
            for stage in self.hparams.data_source:
                if stage in ['predict']:
                    if f'{stage}.json.gz' in existing_files_short and not self.hparams.overwrite_cache:
                        logging.info(f"Found existing {stage}.json.gz in {self.hparams.raw_cache_dir_folder}, Skip")
                    else:
                        logging.info(f"Write {stage}.json.gz to {self.hparams.raw_cache_dir_folder}")
                        for try_num in range(5):
                            try:
                                outfile_path = os.path.join(self.hparams.raw_cache_dir_folder, 
                                    f'{stage}.json.gz')
                                with gzip.open(outfile_path, 'w') as fout:
                                    for file_dict in self.hparams.data_source[stage]:
                                        logging.info(f"Download dvc {file_dict['path']} in {file_dict['repo']} from {self.hparams.data_source_yaml_path} now...")
                                        with dvc.api.open(
                                            path=file_dict['path'],
                                            repo=file_dict['repo'],
                                            rev=file_dict['rev']
                                        ) as f:
                                            if not self.transformer_config.is_encoder_decoder:
                                                logging.warning("encoder-only / decoder-only model development stopped, "
                                                    "please only use encoder-decoder models unless using embedding model")
                                            for l in tqdm(f):
                                                dat = json.loads(l)
                                                # important: text needs to contain exactly one " -> " for encoder-decoder model
                                                try:
                                                    self.transform_write_datum(dat, file_dict, fout)
                                                except Exception as e:
                                                    logging.warning('Skip ' + dat['text'] + f" due to {e}")  
                                logging.info(f"Successfully write {stage}.json.gz to {self.hparams.raw_cache_dir_folder}")
                                break
                            except Exception as e:
                                logging.warning(f'Try {try_num} failed with {e}, rerun Write {stage}.json.gz to {self.hparams.raw_cache_dir_folder}')
                                logging.warning(f"Remove failed {outfile_path}")
                                os.remove(outfile_path)
                                time.sleep(10)

        else:
            logging.info(f"Use cache stored in {self.hparams.raw_cache_dir_folder} for {self.hparams.data_source_yaml_path}")

    def transform_write_datum(self, dat, file_dict, fout):
        in_, _ = dat['text'].split(' -> ')
        fout.write(
            (json.dumps({"text_input": file_dict['task_prefix_input'] + in_}) + '\n').encode('utf-8')
        )
    
    def get_hf_dataset(self):
        logging.info(f"Convert data in {self.hparams.raw_cache_dir_folder} to Huggingface dataset")
        try:
            ds = load_dataset('json', data_files={
                stage: os.path.join(self.hparams.raw_cache_dir_folder, f'{stage}.json.gz') for stage \
                    in self.hparams.data_source if stage in ['predict']
            }, features=Features(text_input = Value(dtype="string", id=None)))

            if 0 in [len(ds[i]) for i in ds]:
                logging.info("Native datasets load failed, use pandas to manually create huggingface dataset")
                ds = DatasetDict({
                    stage: Dataset.from_pandas(
                        pd.read_json(os.path.join(self.hparams.raw_cache_dir_folder, f'{stage}.json.gz'), lines=True)) for stage \
                            in self.hparams.data_source if stage in ['predict']
                })
        except:
            logging.info("Native datasets load failed, use pandas to manually create huggingface dataset")
            ds = DatasetDict({
                stage: Dataset.from_pandas(
                    pd.read_json(os.path.join(self.hparams.raw_cache_dir_folder, f'{stage}.json.gz'), lines=True)) for stage \
                        in self.hparams.data_source if stage in ['predict']
            })
        assert 'predict' in ds
        logging.info(f"hf dataset: {ds}")
        return ds

    def setup(self, stage: str):
        ds = self.get_hf_dataset()
        if not self.transformer_config.is_encoder_decoder:
            logging.warning("encoder-only / decoder-only model development stopped, "
                "please only use encoder-decoder models unless using embedding model")
        def seq2seq_encode(examples):
            input_ = self.tokenizer(examples['text_input'], return_tensors='pt', 
                padding="max_length", truncation=True, 
                max_length=self.hparams.max_length)
            return input_

        ds.set_transform(seq2seq_encode)
        self.ds = ds

    def predict_dataloader(self):
        return utils.data.DataLoader(
            self.ds['predict'], batch_size=self.hparams.batch_size, shuffle=False,
            num_workers=self.hparams.num_workers, pin_memory=True
        )

TASK2DATAMODULE = {
    'clm': LLMData, 
    'dlm': LLM_DenoiseData, 
    'emb': LLM_EmbedData,
    'seqclf': LLM_SeqClassifyData
}

class LLM_MultitaskData(pl.LightningDataModule):
    def __init__(
        self, 
        model_name: str = 't5-base', 
        multitask_dict: dict = {
            "clm": {
                "data_source_yaml_path": "datasets/demo/demo_conditional_lm.yaml",
                "batch_size": 16,
                "max_length": 50,
                "max_length_out": 50 
            },
            "dlm": {
                "data_source_yaml_path": "datasets/demo/demo_denoise_lm.yaml",
                "batch_size": 16,
                "max_length": 50,
                "max_length_out": 50,
                "mask_prob": 0.3,
                "use_ul2": True
            },
            "emb": {
                "data_source_yaml_path": "datasets/demo/demo_embedding.yaml",
                "batch_size": 16,
                "max_length": 50,
                "max_length_out": 50
            },
            "seqclf": {
                "data_source_yaml_path": "datasets/demo/demo_seqclassify.yaml",
                "batch_size": 16,
                "max_length": 50,
                "label_map_file": "datasets/taxonomy/wish_v1.2.1_newtax_allpaths.txt"
            }
        },
        raw_cache_dir: str = os.environ["GENERAL_CACHE"] if "GENERAL_CACHE" in os.environ else None, 
        overwrite_cache: bool = False, 
        predict_on_test: bool = True,
        num_workers: int = multiprocessing.cpu_count(),
        multiple_trainloader_mode: str='max_size_cycle' # or min_size
    ):  
        # check validity of args
        assert multiple_trainloader_mode in ['max_size_cycle', 'min_size']
        # check validity of multitask_dict
        assert len(multitask_dict) > 0, "multitask_dict cannot be empty"
        for task in multitask_dict:
            assert task.split('_')[0] in TASK2DATAMODULE, "task prefix need to be in {}".format(
                TASK2DATAMODULE.keys())
            for p in [ 
                "data_source_yaml_path", 
                "batch_size",
                "max_length", 
                "max_length_out"
            ]:  
                if not task.startswith("seqclf"):
                    assert p in multitask_dict[task], \
                        f"task_dict for {task} needs to contain {p}"
                elif task.startswith("seqclf") and p != "max_length_out":
                    assert p in multitask_dict[task], \
                        f"task_dict for {task} needs to contain {p}"
            if task.startswith('dlm'):
                assert "mask_prob" in multitask_dict[task], \
                    f"task_dict for {task} needs to contain [mask_prob]"
                assert "use_ul2" in multitask_dict[task], \
                    f"task_dict for {task} needs to contain [use_ul2]"
            if task.startswith("seqclf"):
                assert "label_map_file" in multitask_dict[task], \
                    f"task_dict for {task} needs to contain [label_map_file]"

        super().__init__()
        self.datamodules = {}
        for task in multitask_dict:
            if task.startswith('clm') or task.startswith('emb'):
                self.datamodules[task] = TASK2DATAMODULE[task.split('_')[0]](
                    data_source_yaml_path=multitask_dict[task]['data_source_yaml_path'],
                    model_name=model_name,
                    raw_cache_dir=raw_cache_dir,
                    batch_size=multitask_dict[task]['batch_size'],
                    overwrite_cache=overwrite_cache,
                    max_length=multitask_dict[task]['max_length'],
                    predict_on_test=predict_on_test,
                    num_workers=num_workers,
                    max_length_out=multitask_dict[task]['max_length_out']
                )
            elif task.startswith('dlm'):
                self.datamodules[task] = TASK2DATAMODULE[task.split('_')[0]](
                    data_source_yaml_path=multitask_dict[task]['data_source_yaml_path'],
                    model_name=model_name,
                    raw_cache_dir=raw_cache_dir,
                    batch_size=multitask_dict[task]['batch_size'],
                    overwrite_cache=overwrite_cache,
                    max_length=multitask_dict[task]['max_length'],
                    predict_on_test=predict_on_test,
                    num_workers=num_workers,
                    max_length_out=multitask_dict[task]['max_length_out'],
                    mask_prob=multitask_dict[task]['mask_prob'],
                    use_ul2=multitask_dict[task]['use_ul2']
                )
            elif task.startswith('seqclf'):
                self.datamodules[task] = TASK2DATAMODULE[task.split('_')[0]](
                    data_source_yaml_path=multitask_dict[task]['data_source_yaml_path'],
                    model_name=model_name,
                    raw_cache_dir=raw_cache_dir,
                    batch_size=multitask_dict[task]['batch_size'],
                    overwrite_cache=overwrite_cache,
                    max_length=multitask_dict[task]['max_length'],
                    predict_on_test=predict_on_test,
                    num_workers=num_workers,
                    label_map_file=multitask_dict[task]['label_map_file']
                )
            else:
                raise NotImplemented()
        
        self.save_hyperparameters()
    
    def prepare_data(self):
        for task in self.datamodules:
            self.datamodules[task].prepare_data()
    
    def setup(self, stage):
        for task in self.datamodules:
            self.datamodules[task].setup(stage)
    
    def train_dataloader(self):
        outs = {}
        for task in self.datamodules:
            if 'train' in self.datamodules[task].ds:
                outs[task] = utils.data.DataLoader(
                    self.datamodules[task].ds['train'], 
                    batch_size=self.hparams.multitask_dict[task]['batch_size'], 
                    shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True
                )
        return CombinedLoader(outs, mode=self.hparams.multiple_trainloader_mode)

    def val_dataloader(self):
        outs = {}
        for task in self.datamodules:
            if 'val' in self.datamodules[task].ds:
                shuffle = True if task.startswith('emb') else False
                outs[task] = utils.data.DataLoader(
                    self.datamodules[task].ds['val'], 
                    batch_size=self.hparams.multitask_dict[task]['batch_size'], 
                    shuffle=shuffle, num_workers=self.hparams.num_workers, pin_memory=True
                )
        return CombinedLoader(outs, mode=self.hparams.multiple_trainloader_mode)

    def test_dataloader(self):
        outs = {}
        for task in self.datamodules:
            if 'test' in self.datamodules[task].ds:
                shuffle = True if task.startswith('emb') else False
                outs[task] = utils.data.DataLoader(
                    self.datamodules[task].ds['test'], 
                    batch_size=self.hparams.multitask_dict[task]['batch_size'], 
                    shuffle=shuffle, num_workers=self.hparams.num_workers, pin_memory=True
                )
        return CombinedLoader(outs, mode=self.hparams.multiple_trainloader_mode)
    
    def predict_dataloader(self):
        if self.hparams.predict_on_test:
            outs = {}
            for task in self.datamodules:
                if 'test' in self.datamodules[task].ds:
                    outs[task] = utils.data.DataLoader(
                        self.datamodules[task].ds['test'], 
                        batch_size=self.hparams.multitask_dict[task]['batch_size'], 
                        shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True
                    )
            return CombinedLoader(outs, mode=self.hparams.multiple_trainloader_mode)

#%%
class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)