#%%
import os 
import re
import hashlib
import json 
import yaml
import logging
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("asyncssh").disabled = True
import pathlib
import glob
import gzip
import time
import multiprocessing
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from torch import utils
import pandas as pd
from tqdm import tqdm
import dvc.api
from thefuzz import process as fuzz_process
from pytorch_lightning.trainer.supporters import CombinedLoader
from main_utils import (get_transformer, TASK2DATAMODULE, LLM_MultitaskData,
    sortOD, LLM_DenoiseData
)
from datasets import load_dataset, DatasetDict, Dataset
from copy import deepcopy

class JSONListData(pl.LightningDataModule):
    """Flexible Data for arbitrary usage

    Specify arbitrary "input_dict", "output_dict" based on multimodal (or singlemodal) task:

    - for multimodal seqclf (taxonomy classification)

        "input_dict": {
            "template": ("[title start] {title} [title end] "
                "[image start] {image_embedding} [image end]"),
            "task_prefix": "Classify product with image: ",
            "is_multimodal_embedding": ["image_embedding"]
        }
        "output_dict": {
            "template": "{category}"
        }

    - for multimodal clm (taxonomy generation)

        "input_dict": {
            "template": ("[title start] {title} [title end] "
                "[image start] {image_embedding} [image end]"),
            "task_prefix": "Top-down categorize product with image: ",
            "is_multimodal_embedding": ["image_embedding"]
        }
        "output_dict": {
            "template": "{category}"
        }

    - for multimodal clm (title generation)

        "input_dict": {
            "template": "[image start] {image_embedding} [image end]",
            "task_prefix": "Generate title for product with image: ",
            "is_multimodal_embedding": ["image_embedding"]
        }
        "output_dict": {
            "template": "{title}"
        }

    - for multimodal dlm (title denoising)

        "input_dict": {
            "template": ("[title start] {title} [title end] "
                "[image start] {image_embedding} [image end]"),
            "task_prefix": "Denoise product with image: ",
            "is_multimodal_embedding": ["image_embedding"]
        }
        "output_dict": {
            "template": "{title}"
        }

    - for multimodal emb (title+image <> taxonomy)

        "input_dict": {
            "template": ("[title start] {title} [title end] "
                "[image start] {image_embedding} [image end]"),
            "task_prefix": "Embed product with image: ",
            "is_multimodal_embedding": ["image_embedding"]
        }
        "output_dict": {
            "template": "{category}",
            "task_prefix": "Embed taxonomy: "
        }

    - for multimodal emb (title <> image)

        "input_dict": {
            "template": "{title}",
            "task_prefix": "Embed product: "
        }
        "output_dict": {
            "template": "[image start] {image_embedding} [image end]",
            "task_prefix": "Embed image: ",
            "is_multimodal_embedding": ["image_embedding"]
        }
    
    - for multimodal emb (title+image <> query)

        "input_dict": {
            "template": ("[title start] {title} [title end] "
                "[image start] {image_embedding} [image end]"),
            "task_prefix": "Embed product with image: ",
            "is_multimodal_embedding": ["image_embedding"]
        }
        "output_dict": {
            "template": "{query}",
            "task_prefix": "Embed query: ",
        }

    - for multimodal manual emb (title+image <> query, but with specified relevance score)

        "input_dict": {
            "template": ("[title start] {title} [title end] "
                "[image start] {image_embedding} [image end]"),
            "task_prefix": "Embed product with image: ",
            "is_multimodal_embedding": ["image_embedding"]
        }
        "output_dict": {
            "template": "{query}{relevance}",
            "task_prefix": "Embed query: ",
            "is_manual": ["relevance"]
        }

    - for text-only seqclf (taxonomy classification for query)

        "input_dict": {
            "template": "{query}",
            "task_prefix": "Classify query: "
        }
        "output_dict": {
            "template": "{category}"
        }

    - for text-only seqclf (taxonomy classification for product)

        "input_dict": {
            "template": "{title}",
            "task_prefix": "Classify product: "
        }
        "output_dict": {
            "template": "{category}"
        }
    
    - for text-only seqclf (taxonomy classification for product, with description)

        "input_dict": {
            "template": ("[title start] {title} [title end] "
                "[description start] {description} [description end]"),
            "task_prefix": "Classify product with description: "
        }
        "output_dict": {
            "template": "{category}"
        }

    - for text-only clm (taxonomy generation for query)

        "input_dict": {
            "template": ("{query}"),
            "task_prefix": "Top-down categorize query: "
        }
        "output_dict": {
            "template": "{category}"
        }

    - for text-only clm (taxonomy generation for product)

        "input_dict": {
            "template": ("{title}"),
            "task_prefix": "Top-down categorize product: "
        }
        "output_dict": {
            "template": "{category}"
        }

    - for text-only clm (description generation)

        "input_dict": {
            "template": ("{title}"),
            "task_prefix": "Describe product: "
        }
        "output_dict": {
            "template": "{description}"
        }

    - for text-only dlm (title denoising)

        "input_dict": {
            "template": ("{title}"),
            "task_prefix": "Denoise product: "
        }
        "output_dict": {
            "template": "{title}"
        }

    - for text-only emb (title <> taxonomy)

        "input_dict": {
            "template": "{title}",
            "task_prefix": "Embed product: "
        }
        "output_dict": {
            "template": "{category}",
            "task_prefix": "Embed taxonomy: "
        }

    - for text-only emb (title <> co-purchased title)

        "input_dict": {
            "template": "{title1}",
            "task_prefix": "Embed co-purchased product: "
        }
        "output_dict": {
            "template": "{title2}",
            "task_prefix": "Embed co-purchased product: "
        }

    - for text-only emb (title <> query)

        "input_dict": {
            "template": "{title}",
            "task_prefix": "Embed product: "
        }
        "output_dict": {
            "template": "{query}",
            "task_prefix": "Embed query: "
        }
    
    - for text-only manual emb (title <> query, but with specified relevance score)

        "input_dict": {
            "template": "{title}",
            "task_prefix": "Embed product: "
        }
        "output_dict": {
            "template": "{query}{relevance}",
            "task_prefix": "Embed query: ",
            "is_manual": ["relevance"]
        }

    """
    def __init__(
        # mandatory args
        self, llm_type: str=None, data_source_yaml_path: str=None, input_dict: dict=None, output_dict: dict=None,
        # nuisance params
        data_source_type: str = 'dvc',
        model_name: str = 't5-base', 
        raw_cache_dir: str = os.environ["GENERAL_CACHE"] if "GENERAL_CACHE" in os.environ else None, 
        batch_size: int = 16, 
        overwrite_cache: bool = False, 
        max_length: int = 50,
        predict_on_test: bool = True,
        predict_on_trainval: str = None,
        num_workers: int = multiprocessing.cpu_count(),
        max_length_out: int = 50,
        force_download_hfdata: bool = False,
        # seqclf
        label_map_file: str = None,
        label_type: str = None,
        # denoise
        mask_prob: float = 0.3,
        use_ul2: bool = False,
        # transform
        transform_dict: dict = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.llm_type is not None, "llm_type cannot be None"
        assert self.hparams.data_source_yaml_path is not None, "data_source_yaml_path cannot be None"
        assert self.hparams.data_source_type in ['dvc', 'local'], "data_source_type needs to be valid"
        assert self.hparams.raw_cache_dir is not None, "raw_cahce_dir cannot be None"
        assert self.hparams.input_dict is not None, "input_dict cannot be None"
        assert self.hparams.output_dict is not None, "output_dict cannot be None"
        if self.hparams.predict_on_test:
            assert self.hparams.predict_on_trainval is None 
        if self.hparams.predict_on_trainval is not None: 
            self.hparams.predict_on_trainval in ['train', 'val']
        # check validity of llm_type
        assert any(
            self.hparams.llm_type.startswith(i) for i in TASK2DATAMODULE
        )
        # check validity of input_dict and output_dict
        assert "template" in self.hparams.input_dict
        assert "task_prefix" in self.hparams.input_dict
        assert "template" in self.hparams.output_dict
        if self.hparams.llm_type.startswith('emb'):
            assert "task_prefix" in self.hparams.output_dict
        else:
            assert "task_prefix" not in self.hparams.output_dict

        if "is_multimodal_embedding" in self.hparams.input_dict:
            assert len(self.hparams.input_dict["is_multimodal_embedding"]) > 0
            for k in self.hparams.input_dict["is_multimodal_embedding"]:
                assert f"{{{k}}}" in self.hparams.input_dict["template"]
        if "is_multimodal_embedding" in self.hparams.output_dict:
            assert self.hparams.llm_type.startswith("emb")
            assert len(self.hparams.output_dict["is_multimodal_embedding"]) > 0
            for k in self.hparams.output_dict["is_multimodal_embedding"]:
                assert f"{{{k}}}" in self.hparams.output_dict["template"]
        if "is_manual" in self.hparams.output_dict:
            assert self.hparams.llm_type.startswith("emb")
            assert len(self.hparams.output_dict["is_manual"]) > 0
            self.relevance_key = self.hparams.output_dict["is_manual"]
            assert len(self.relevance_key) == 1, "multiple relevance key is not supported now"

        # load tokenizer and config for transformer
        self.transformer_config, self.tokenizer = get_transformer(
            self.hparams.model_name, return_model=False, **kwargs)
        # save special tokens
        self.local_additional_special_tokens_list = deepcopy(self.tokenizer.additional_special_tokens)

        # setup seqclf-specific params
        if self.hparams.llm_type.startswith('seqclf'):
            assert self.hparams.label_map_file is not None, "seqclf task needs to specify label_map_file"
            # TODO: add more label_type
            assert self.hparams.label_type is not None and self.hparams.label_type in ["taxonomy"], \
                "seqclf task needs to specify valid label_type"
            self.label_map = {}
            with open(self.hparams.label_map_file, 'r') as f:
                for l in f:
                    l = l.replace('\n', '').strip()
                    if len(l):
                        self.label_map[l] = len(self.label_map)
            self.label_list = sorted([i for i in self.label_map.items()], key=lambda x: x[1])
            self.label_list = [i[0] for i in self.label_list]
            self.match_label_map = {}
            self.label_key = re.findall(r'{(.*?)}', self.hparams.output_dict['template'])
            assert len(self.label_key) == 1, "multiple label key is not supported now"

        # setup dlm-specific params
        if self.hparams.llm_type.startswith('dlm'):
            self.input_denoise_key = re.findall(r'{(.*?)}', self.hparams.input_dict["template"])
            self.output_denoise_key = re.findall(r'{(.*?)}', self.hparams.output_dict["template"])
            assert len(set(self.input_denoise_key)) == len(self.input_denoise_key) and \
                len(self.input_denoise_key) > 0, "denoising can only take unique input keys (at least 1)"
            assert len(set(self.output_denoise_key)) == len(self.output_denoise_key) and \
                len(self.output_denoise_key) > 0, "denoising can only take unique output keys (at least 1)"
            assert all(i in self.input_denoise_key for i in self.output_denoise_key), "denoising output keys need to be in input keys"

        
        # get unique dataset identifier with unique path
        data_source_dict = yaml.safe_load(open(self.hparams.data_source_yaml_path, 'r'))
        data_source_dict['is_encoder_decoder'] = self.transformer_config.is_encoder_decoder is True
        data_source_dict['is_jsonl_datamodule'] = True
        data_source_dict['llm_type_is_seqclf'] = self.hparams.llm_type.startswith('seqclf')
        if self.hparams.llm_type.startswith('seqclf'):
            data_source_dict['output_dict'] = output_dict
        self.hparams.data_source = sortOD(data_source_dict)
        self.hparams.data_hash = hashlib.md5(
            json.dumps(self.hparams.data_source).encode('utf-8')
        ).hexdigest()
        self.hparams.raw_cache_dir_folder = os.path.join(self.hparams.raw_cache_dir, 
            self.hparams.data_hash)
        pathlib.Path(self.hparams.raw_cache_dir_folder).mkdir(parents=True, exist_ok=True)
    
    def get_label_matched_text(self, text):
        if text in self.label_map:
            return text 
        elif text in self.match_label_map:
            return self.match_label_map[text]
        else:
            text_match = fuzz_process.extractOne(text, self.label_list)
            logging.warning(f"{text} not in label_map, matched to {text_match}")
            self.match_label_map[text] = text_match[0]
            logging.warning(f"match_label_map grow to {len(self.match_label_map)}")
            return text_match[0]

    def text2label(self, text, label_type):
        """Convert plain text label to numerical label"""
        assert isinstance(text, str)
        if label_type == "taxonomy":
            label = [0] * len(self.label_map)
            textlist = text.split(" > ")
            l_s = textlist[0]
            label[self.label_map[l_s]] = 1
            for l in textlist[1:]:
                l_s = l_s + " > " + l
                label[self.label_map[l_s]] = 1
            return label
        else:
            raise NotImplemented()

    def transform_example_content(self, x, transform_type):
        if transform_type == "taxonomy":
            return " > ".join([i.strip().lower() for i in x])
        elif transform_type == "eval":
            return eval(x)
        else:
            raise NotImplemented()

    def prepare_data(self):
        
        def helper(f):
            if not self.transformer_config.is_encoder_decoder:
                logging.warning("encoder-only / decoder-only model development stopped, "
                    "please only use encoder-decoder models unless using embedding model")
            for l in tqdm(f):
                dat = json.loads(l)
                try:
                    if self.hparams.llm_type.startswith('seqclf'):
                        for lk in self.label_key:
                            if self.hparams.label_type == "taxonomy":
                                labels = self.get_label_matched_text(" > ".join([i.strip().lower() for i in dat[lk]]))
                                assert f"labels_{lk}" not in dat
                                assert isinstance(labels, str)
                                dat[f"labels_{lk}"] = labels
                            else:
                                raise NotImplemented()
                        fout.write((json.dumps({"json_content": dat}) + '\n').encode('utf-8'))
                    else:
                        fout.write((json.dumps({"json_content": dat}) + '\n').encode('utf-8'))
                except Exception as e:
                    logging.warning('Skip ' + dat + f" due to {e}")

        existing_files = glob.glob(self.hparams.raw_cache_dir_folder + '/*.json.gz')
        existing_files_short = [i.split('/')[-1] for i in existing_files]
        if len(existing_files) != len([i for i in self.hparams.data_source if i in ['train', 'val', 'test', 'predict']]) or \
                self.hparams.overwrite_cache:
            if self.hparams.data_source_type == 'dvc':
                logging.info(f"loading {self.hparams.data_source_yaml_path} from dvc")
            elif self.hparams.data_source_type == 'local':
                logging.info(f"loading {self.hparams.data_source_yaml_path} from local")

            for stage in self.hparams.data_source:
                if stage in ['train', 'val', 'test', 'predict']:
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
                                        if self.hparams.data_source_type == 'dvc':
                                            logging.info(f"Download dvc {file_dict['path']} in {file_dict['repo']} from {self.hparams.data_source_yaml_path} now...")
                                        
                                            with dvc.api.open(
                                                path=file_dict['path'],
                                                repo=file_dict['repo'],
                                                rev=file_dict['rev']
                                            ) as f:
                                                helper(f)
                                        elif self.hparams.data_source_type == 'local':
                                            logging.info(f"Download local {file_dict['path']} from {self.hparams.data_source_yaml_path} now...")
                                            if file_dict['path'].endswith(".gz"):
                                                with gzip.open(
                                                    file_dict['path'], 'r'
                                                ) as f:
                                                    helper(f)
                                            else:
                                                with open(
                                                    file_dict['path'], 'r'
                                                ) as f:
                                                    helper(f)

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
            ds = DatasetDict({
                stage: load_dataset('json', download_mode='force_redownload' if self.hparams.force_download_hfdata else 'reuse_dataset_if_exists', 
                data_files=os.path.join(
                    self.hparams.raw_cache_dir_folder, f'{stage}.json.gz'), split='train') for stage \
                        in self.hparams.data_source if stage in ['train', 'val', 'test', 'predict']
            })

            if 0 in [len(ds[i]) for i in ds]:
                logging.info("Native datasets load failed, use pandas to manually create huggingface dataset")
                ds = DatasetDict({
                    stage: Dataset.from_pandas(
                        pd.read_json(os.path.join(self.hparams.raw_cache_dir_folder, f'{stage}.json.gz'), lines=True)) for stage \
                            in self.hparams.data_source if stage in ['train', 'val', 'test', 'predict']
                })
        except Exception as e:
            logging.warning(e)
            logging.info("Native datasets load failed, use pandas to manually create huggingface dataset")
            ds = DatasetDict({
                stage: Dataset.from_pandas(
                    pd.read_json(os.path.join(self.hparams.raw_cache_dir_folder, f'{stage}.json.gz'), lines=True)) for stage \
                        in self.hparams.data_source if stage in ['train', 'val', 'test', 'predict']
            })
        logging.info(f"hf dataset for {self.hparams.data_source_yaml_path}: {ds}")
        return ds
    
    def setup(self, stage: str):
        ds = self.get_hf_dataset()
        if self.transformer_config.is_encoder_decoder:

            def process_single_example(example):
                assert isinstance(example, dict)
                example = deepcopy(example)
                if self.hparams.transform_dict is not None:
                    for k in self.hparams.transform_dict:
                        example[k] = self.transform_example_content(example[k], self.hparams.transform_dict[k])
                processed_example = {}
                input_template = deepcopy(self.hparams.input_dict["template"])
                # input for non-dlm
                if self.hparams.llm_type.startswith('seqclf') or \
                    self.hparams.llm_type.startswith('clm') or \
                    self.hparams.llm_type.startswith('emb'):
                    
                    if "is_multimodal_embedding" in self.hparams.input_dict:
                        for ind, k in enumerate(self.hparams.input_dict["is_multimodal_embedding"]):
                            input_template = input_template.replace(
                                f"{{{k}}}",
                                self.local_additional_special_tokens_list[-(1+ind)] # use special tokens from tail first
                            )
                            processed_example[f"input_multimodal_embedding_{ind}"] = example[k] if isinstance(example[k], list) else eval(example[k])
                            
                    processed_example["input_text"] = (self.hparams.input_dict["task_prefix"] + input_template.format(**example)).strip()

                elif self.hparams.llm_type.startswith('dlm'):
                    # input and output for dlm
                    denoise_inputs = {}
                    denoise_outputs = {}
                    start_at_special_ind = 0
                    for k in self.output_denoise_key:
                        input_text = example[k]
                        # leave space for multimodal placeholder
                        if "is_multimodal_embedding" in self.hparams.input_dict:
                            in_out_c_ = LLM_DenoiseData.mask_input(input_text, self.hparams.mask_prob, self.hparams.use_ul2, 
                                self.local_additional_special_tokens_list[start_at_special_ind:-len(self.hparams.input_dict["is_multimodal_embedding"])], 
                                use_task_prefix=False, return_num_special_tokens_inserted=True)
                        else:
                            in_out_c_ = LLM_DenoiseData.mask_input(input_text, self.hparams.mask_prob, self.hparams.use_ul2, 
                                self.local_additional_special_tokens_list[start_at_special_ind:], use_task_prefix=False,
                                return_num_special_tokens_inserted=True)
                        if in_out_c_ is None:
                            denoise_inputs[k] = input_text
                            # return <pad> if unknown (HACK but the propoer way to handle is too complex)
                            # this basically discards the example due to logic in "def process_multiple_examples"
                            denoise_outputs[k] = "<pad>" 
                        else:
                            denoise_inputs[k] = in_out_c_[0]
                            denoise_outputs[k] = in_out_c_[1]
                            start_at_special_ind += in_out_c_[2]
                    if "is_multimodal_embedding" in self.hparams.input_dict:
                         for ind, k in enumerate(self.hparams.input_dict["is_multimodal_embedding"]):
                            input_template = input_template.replace(
                                f"{{{k}}}",
                                self.local_additional_special_tokens_list[-(1+ind)] # use special tokens from tail first
                            )
                            processed_example[f"input_multimodal_embedding_{ind}"] = example[k] if isinstance(example[k], list) else eval(example[k])
                    for k in self.input_denoise_key:
                        if k not in self.output_denoise_key and k not in self.hparams.input_dict["is_multimodal_embedding"]:
                            denoise_inputs[k] = example[k]
                    processed_example["input_text"] = (self.hparams.input_dict["task_prefix"] + input_template.format(**denoise_inputs)).strip()
                    processed_example["output_text"] = ""
                    for ind, k in enumerate(self.output_denoise_key):
                        if ind == 0:
                            processed_example["output_text"] += denoise_outputs[k]
                        else:
                            # if denoising on multiple chunks of input, skip initial special token
                            processed_example["output_text"] += " ".join(denoise_outputs[k].split(" ")[1:])
                else:
                    raise NotImplemented()

                # output for non-dlm
                if not self.hparams.llm_type.startswith('dlm'):
                    output_template = deepcopy(self.hparams.output_dict["template"])

                    if self.hparams.llm_type.startswith('seqclf'):
                        for lk in self.label_key:
                            processed_example[f"output_labels_{lk}"] = self.text2label(example[f"labels_{lk}"], self.hparams.label_type)

                    elif self.hparams.llm_type.startswith('clm'):
                        processed_example["output_text"] = output_template.format(**example).strip()

                    elif self.hparams.llm_type.startswith('emb'):
                        if "is_manual" in self.hparams.output_dict:
                            for k in self.relevance_key:
                                output_template = output_template.replace(f"{{{k}}}", "")
                                assert isinstance(example[k], int) or isinstance(example[k], float)
                                processed_example["output_labels_{k}"] = [float(example[k])]
                        if "is_multimodal_embedding" in self.hparams.output_dict:
                            for ind, k in enumerate(self.hparams.output_dict["is_multimodal_embedding"]):
                                output_template = output_template.replace(
                                    f"{{{k}}}",
                                    self.local_additional_special_tokens_list[-(1+ind)] # use special tokens from tail first
                                )
                                processed_example[f"output_multimodal_embedding_{ind}"] = example[k] if isinstance(example[k], list) else eval(example[k])
                        processed_example["output_text"] = (self.hparams.output_dict["task_prefix"] + output_template.format(**example)).strip()

                return processed_example

            def process_multiple_examples(examples):
                processed_examples = [process_single_example(i) if isinstance(i, dict) else process_single_example(json.loads(i)) for i in examples['json_content']]
                # all task must have "input_text"
                batch = self.tokenizer([i["input_text"] for i in processed_examples], return_tensors='pt', 
                    padding="max_length", truncation=True, max_length=self.hparams.max_length)
                # clm, dlm must have "output_text"
                if self.hparams.llm_type.startswith('clm') or self.hparams.llm_type.startswith('dlm'):
                    output_ = self.tokenizer([i['output_text'] for i in processed_examples], return_tensors='pt', 
                        padding="max_length", truncation=True, max_length=self.hparams.max_length_out)
                    labels = output_.input_ids
                    labels[labels == self.tokenizer.pad_token_id] = -100
                    batch['labels'] = labels
                # emb must have "output_text"
                elif self.hparams.llm_type.startswith('emb'):
                    output_ = self.tokenizer([i['output_text'] for i in processed_examples], return_tensors='pt', 
                        padding="max_length", truncation=True, max_length=self.hparams.max_length_out)
                    for i in output_:
                        batch[f'output_{i}'] = output_[i]
                # seqclf must have "output_labels_*"
                elif self.hparams.llm_type.startswith('seqclf'):

                    if len(self.label_key) == 1:
                        k = self.label_key[0]
                        batch['labels'] = torch.FloatTensor([i[f"output_labels_{k}"] for i in processed_examples])
                    else:
                        raise NotImplemented()
                        # for k in self.label_key:
                        #     batch['labels_{k}'] = torch.FloatTensor([i[f"output_labels_{k}"] for i in processed_examples])

                if "is_multimodal_embedding" in self.hparams.input_dict:
                    for ind, k in enumerate(self.hparams.input_dict["is_multimodal_embedding"]):
                        n = f"input_multimodal_embedding_{ind}"
                        batch[n] = torch.FloatTensor([i[n] for i in processed_examples])
                if "is_multimodal_embedding" in self.hparams.output_dict:
                    for ind, k in enumerate(self.hparams.output_dict["is_multimodal_embedding"]):
                        n = f"output_multimodal_embedding_{ind}"
                        batch[n] = torch.FloatTensor([i[n] for i in processed_examples])
                return batch

            ds.set_transform(process_multiple_examples)
        else:
            raise NotImplemented()

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
            logging.info(f"Predict using test dataset in {self.hparams.data_source_yaml_path}")
            return utils.data.DataLoader(
                self.ds['test'], batch_size=self.hparams.batch_size, shuffle=False,
                num_workers=self.hparams.num_workers, pin_memory=True
            )
        elif self.hparams.predict_on_trainval is not None and self.hparams.predict_on_trainval in self.ds:
            logging.info(f"Predict using {self.hparams.predict_on_trainval} dataset in {self.hparams.data_source_yaml_path}")
            return utils.data.DataLoader(
                self.ds[self.hparams.predict_on_trainval], batch_size=self.hparams.batch_size, shuffle=False,
                num_workers=self.hparams.num_workers, pin_memory=True
            )
        elif "predict" in self.ds:
            logging.info(f"Predict using predict dataset in {self.hparams.data_source_yaml_path}")
            return utils.data.DataLoader(
                self.ds['predict'], batch_size=self.hparams.batch_size, shuffle=False,
                num_workers=self.hparams.num_workers, pin_memory=True
            )
        else:
            logging.info(f"No predict dataset available in {self.hparams.data_source_yaml_path}")

class LLM_MultitaskMultiModalData(pl.LightningDataModule):
    def __init__(
        self, 
        model_name: str = 't5-base', 
        multitask_dict: dict = {
            "clm_multimodal_clip2wishtitle": {
                "data_source_yaml_path": "/workspaces/query_understanding_model/datasets/demo_local/demo_local_multimodal_full.yaml",
                "batch_size": 16,
                "max_length": 50,
                "max_length_out": 50,
                "llm_type": "clm",
                "input_dict": {
                    "template": "[image start] {img_embedding} [image end]",
                    "task_prefix": "Generate title for product with image: ",
                    "is_multimodal_embedding": ["img_embedding"]
                },
                "output_dict": {
                    "template": "{title}"
                },
                "data_source_type": "local"
            },
            "dlm_multimodal_wishtitlewclip": {
                "data_source_yaml_path": "/workspaces/query_understanding_model/datasets/demo_local/demo_local_multimodal_full.yaml",
                "batch_size": 16,
                "max_length": 50,
                "max_length_out": 50,
                "llm_type": "dlm",
                "input_dict": {
                    "template": "[title start] {title} [title end] [image start] {img_embedding} [image end]",
                    "task_prefix": "Denoise product with image: ",
                    "is_multimodal_embedding": ["img_embedding"]
                },
                "output_dict": {
                    "template": "{title}"
                },
                "data_source_type": "local"
            },
            "seqclf_multimodal_wishtitlewclip2pseudov121tax": {
                "data_source_yaml_path": "/workspaces/query_understanding_model/datasets/demo_local/demo_local_multimodal_full.yaml",
                "batch_size": 16,
                "max_length": 50,
                "max_length_out": 50,
                "label_map_file": "/workspaces/query_understanding_model/datasets/taxonomy/wish_v1.2.1_newtax_allpaths.txt",
                "label_type": "taxonomy", 
                "llm_type": "seqclf",
                "input_dict": {
                    "template": "[title start] {title} [title end] [image start] {img_embedding} [image end]",
                    "task_prefix": "Classify product with image: ",
                    "is_multimodal_embedding": ["img_embedding"]
                },
                "output_dict": {
                    "template": "{pseudo_category}"
                },
                "transform_dict": {
                    "pseudo_category": "taxonomy"
                },
                "data_source_type": "local"
            },
            "seqclf_singlemodal_alititle2v121tax": {
                "data_source_yaml_path": "/workspaces/query_understanding_model/datasets/demo_local/demo_local_multimodal_full.yaml",
                "batch_size": 16,
                "max_length": 50,
                "max_length_out": 50,
                "label_map_file": "/workspaces/query_understanding_model/datasets/taxonomy/wish_v1.2.1_newtax_allpaths.txt",
                "label_type": "taxonomy", 
                "llm_type": "seqclf",
                "input_dict": {
                    "template": ("{title}"),
                    "task_prefix": "Classify product: "
                },
                "output_dict": {
                    "template": "{category}"
                },
                "transform_dict": {
                    "category": "taxonomy"
                },
                "data_source_type": "local"
            },
            "emb_singlemodal_wishquery2googletitle": {
                "data_source_yaml_path": "/workspaces/query_understanding_model/datasets/demo_local/demo_local_multimodal_full.yaml",
                "batch_size": 16,
                "max_length": 50,
                "max_length_out": 50,
                "llm_type": "emb",
                "input_dict": {
                    "template": "{title}",
                    "task_prefix": "Embed product: "
                },
                "output_dict": {
                    "template": "{query}",
                    "task_prefix": "Embed query: "
                },
                "data_source_type": "local"
            },
            "emb_singlemodal_amaquery2amatitle_manual": {
                "data_source_yaml_path": "/workspaces/query_understanding_model/datasets/demo_local/demo_local_multimodal_full.yaml",
                "batch_size": 16,
                "max_length": 50,
                "max_length_out": 50,
                "llm_type": "emb",
                "input_dict": {
                    "template": "{title}",
                    "task_prefix": "Embed product: "
                },
                "output_dict": {
                    "template": "{query}{relevance}",
                    "task_prefix": "Embed query: ",
                    "is_manual": ["relevance"]
                },
                "data_source_type": "local"
            }
        },
        raw_cache_dir: str = os.environ["GENERAL_CACHE"] if "GENERAL_CACHE" in os.environ else None, 
        overwrite_cache: bool = False, 
        force_download_hfdata: bool = False,
        predict_on_test: bool = True,
        predict_on_trainval: str = None,
        num_workers: int = multiprocessing.cpu_count(),
        multiple_trainloader_mode: str='max_size_cycle' # or min_size
    ):  
        # check validity of args
        assert multiple_trainloader_mode in ['max_size_cycle', 'min_size']
        # check validity of multitask_dict
        assert multitask_dict is not None and len(multitask_dict) > 0, "multitask_dict cannot be empty"
        # check validity of predict_on_trainval 
        if predict_on_test:
            assert predict_on_trainval is None 
        if predict_on_trainval is not None: 
            assert predict_on_trainval in ['train', 'val']

        super().__init__()
        self.datamodules = {}

        for task in multitask_dict:
            assert task.split('_')[0] in TASK2DATAMODULE, "task prefix need to be in {}".format(
                TASK2DATAMODULE.keys())
            for general_param in ["model_name", "raw_cache_dir", "overwrite_cache", "predict_on_test",
                    "num_workers"]:
                assert general_param not in multitask_dict[task], f"general_param {general_param} should not be in multitask_dict"
            
            self.datamodules[task] = JSONListData(
                model_name=model_name,
                raw_cache_dir=raw_cache_dir,
                overwrite_cache=overwrite_cache,
                force_download_hfdata=force_download_hfdata,
                predict_on_test=predict_on_test,
                num_workers=num_workers,
                **multitask_dict[task]
            )
        
        self.save_hyperparameters()

    def prepare_data(self):
        for task in self.datamodules:
            self.datamodules[task].prepare_data()
    
    def setup(self, stage):
        for task in self.datamodules:
            self.datamodules[task].setup(stage)
    
    @rank_zero_only
    def log_sample_batch(self, split, task):
        logging.info(f"\n\n==> Example {split} sample for {task}: ")
        sample_batch = self.datamodules[task].ds[split][:1]
        try:
            for i in sample_batch:
                if "input_ids" in i or "labels" in i:
                    tmp = deepcopy(sample_batch[i])
                    tmp[tmp<0] = 0
                    logging.info("> " + i + ": " + self.datamodules[task].tokenizer.batch_decode(tmp)[0])
                else:
                    logging.info("> " + i + ": " + str(sample_batch[i]))
        except Exception as e:
            logging.error(e)
            logging.info(sample_batch)

    def train_dataloader(self):
        outs = {}
        for task in self.datamodules:
            if 'train' in self.datamodules[task].ds:
                outs[task] = utils.data.DataLoader(
                    self.datamodules[task].ds['train'], 
                    batch_size=self.hparams.multitask_dict[task]['batch_size'], 
                    shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True
                )
                self.log_sample_batch('train', task)
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
                self.log_sample_batch('val', task)
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
                self.log_sample_batch('test', task)
        return CombinedLoader(outs, mode=self.hparams.multiple_trainloader_mode)
    
    def predict_dataloader(self):
        key = "predict"
        if self.hparams.predict_on_test:
            key = "test"
        elif self.hparams.predict_on_trainval is not None:
            key = self.hparams.predict_on_trainval
        outs = {}
        for task in self.datamodules:
            if key in self.datamodules[task].ds:
                outs[task] = utils.data.DataLoader(
                    self.datamodules[task].ds[key], 
                    batch_size=self.hparams.multitask_dict[task]['batch_size'], 
                    shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True
                )
                self.log_sample_batch(key, task)
        return CombinedLoader(outs, mode=self.hparams.multiple_trainloader_mode)