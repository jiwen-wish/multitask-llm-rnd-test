#%%
import logging
logging.getLogger().setLevel(logging.INFO)

import torch
from torch import nn
from main_embedding import LLM_Embed
from main_utils import TASK2DATAMODULE
from main_utils_multimodal import LLM_MultitaskMultiModalData
from pytorch_lightning.cli import LightningCLI
from copy import deepcopy
from typing import Any, Mapping, Dict


#%%
class LLM_MultitaskMultimodal(LLM_Embed):
    """Unfortunately needs to rewrite large chunks of LLM_Embed, LLM_SeqClassify due to this class being way more general, and would break inheritance
    However, subclassing LLM_Embed makes most sense compared to subclassing LLM_SeqCLassify (__init__ of LLM_SeqClassify is too rigid)
    """
    def __init__(self, 
            # basics
            model_name: str = 't5-base', weight_decay: float = 0.1, 
            learning_rate: float = 1e-4, lr_scheduler_max_steps: int = None, 
            lr_scheduler_type: str = None, lr_scheduler_num_warmup_steps: int = None,
            load_weights_only: bool=False,
            # seqclf / embed lm (abuse LLM_Embed base class a bit since LLM_MultitaskMultimodal should handle both auto-in-batch-negative and manual emb loss)
            distance_func: str='cosine', loss_type: str='cross-entropy', manual_loss_type: str='manual_mse', 
            margin: float=None, # margin not used in manual mode
            hidden_states_type: str='encoder-last', # the same hidden_states_type is used across emb and seqclf for simplicity
            add_simcse: bool=True, # simcse only in auto emb mode
            # general multitask config
            auto_task_weight: bool=False, # Paper: https://arxiv.org/pdf/1705.07115v3.pdf
            # multi task specs
            multitask_specs_dict: dict = {
                "clm_multimodal_clip2wishtitle": {
                    "multimodal_embedding": {
                        "input": [{
                            "key": "img_embedding", 
                            "proj_head": "proj_head",
                        }]
                    }
                },
                "dlm_multimodal_wishtitlewclip": {
                    "multimodal_embedding": {
                        "input": [{
                            "key": "img_embedding", 
                            "proj_head": "proj_head",
                        }]
                    }
                },
                "seqclf_multimodal_wishtitlewclip2pseudov121tax": {
                    "multimodal_embedding": {
                        "input": [{
                            "key": "img_embedding", 
                            "proj_head": "proj_head",
                        }]
                    },
                    "specs": {
                        "clf_head": "clf_head",
                        "clf_weight_type": "ancestor-high",
                        "label_map_file": "/workspaces/multitask-llm-rnd/modelling/datasets/taxonomy/wish_v1.2.1_newtax_allpaths.txt",
                        "label_type": "taxonomy", 
                    }
                },
                "seqclf_singlemodal_alititle2v121tax": {
                    "specs": {
                        "clf_head": "clf_head",
                        "clf_weight_type": "ancestor-high",
                        "label_map_file": "/workspaces/multitask-llm-rnd/modelling/datasets/taxonomy/wish_v1.2.1_newtax_allpaths.txt",
                        "label_type": "taxonomy", 
                    }
                },
                "emb_singlemodal_wishquery2googletitle": None,
                "emb_singlemodal_amaquery2amatitle_manual": {
                    "specs": {
                        "is_manual": ["relevance"]
                    }
                }
            },
            # multi task head specs
            head_dict: dict = {
                "proj_head": {
                    "purpose": "projection",
                    "type": "linear",
                    "in_features": 768,
                    "out_features": 768
                },
                "clf_head": {
                    "purpose": "seqclf",
                    "type": "linear",
                    "in_features": 768,
                    "out_features": 6037
                }
            },
            # other kwargs
            **kwargs
        ):

        super().__init__(
            model_name = model_name, 
            weight_decay = weight_decay, 
            learning_rate = learning_rate, 
            lr_scheduler_max_steps = lr_scheduler_max_steps, 
            lr_scheduler_type = lr_scheduler_type, 
            lr_scheduler_num_warmup_steps = lr_scheduler_num_warmup_steps,
            distance_func = distance_func,
            loss_type = loss_type,
            manual_loss_type=manual_loss_type,
            margin = margin,
            hidden_states_type = hidden_states_type,
            add_simcse = add_simcse,
            auto_task_weight = auto_task_weight,
            multitask_specs_dict=multitask_specs_dict,
            head_dict=head_dict,
            **kwargs
        )

        assert multitask_specs_dict is not None
        # validate multitask_specs_dict and head_dict
        for task in multitask_specs_dict:
            assert task.split('_')[0] in TASK2DATAMODULE
            # multimodal
            if multitask_specs_dict[task] is not None and "multimodal_embedding" in multitask_specs_dict[task]:
                for k in multitask_specs_dict[task]["multimodal_embedding"]:
                    assert k in ["input", "output"]
                    for ind in range(len(multitask_specs_dict[task]["multimodal_embedding"][k])):
                        assert set(multitask_specs_dict[task]["multimodal_embedding"][k][ind]) == set(["key", "proj_head"])
                        assert multitask_specs_dict[task]["multimodal_embedding"][k][ind]["proj_head"] in head_dict
            # seqclf
            if task.startswith('seqclf'):
                assert multitask_specs_dict[task] is not None
                assert "specs" in multitask_specs_dict[task]
                assert set(multitask_specs_dict[task]["specs"]) == set(["clf_head", "clf_weight_type", "label_map_file", "label_type"])
                assert multitask_specs_dict[task]["specs"]["clf_head"] in head_dict
            # emb
            if task.startswith('emb'):
                if multitask_specs_dict[task] is not None:
                    assert len(multitask_specs_dict[task]["specs"]["is_manual"]) == 1, "only able to specify one relevance score for manual embed"

        if self.transformer_config.is_encoder_decoder:
            emb_dim = self.transformer_config.d_model
        else:
            emb_dim = self.transformer_config.hidden_size

        if head_dict is not None: 
            for k in head_dict:
                assert set(head_dict[k]) == set(["type", "in_features", "out_features", "purpose"])
                if head_dict[k]["purpose"] == "projection":
                    assert head_dict[k]["out_features"] == emb_dim

        # auto task weight
        # Paper: https://arxiv.org/pdf/1705.07115v3.pdf
        # Code: https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example-pytorch.ipynb
        if self.hparams.auto_task_weight:
            self.log_vars = nn.ParameterDict({
                task: nn.Parameter(torch.scalar_tensor(1)) for task \
                    in self.hparams.multitask_specs_dict
            })

        # init clf and proj heads if any
        if self.hparams.head_dict is not None and len(self.hparams.head_dict) > 0:
            for i in self.hparams.head_dict:
                assert self.hparams.head_dict[i]["type"] == "linear", "only linear heads is implemented now"
                # hard coding unimportant params like dropout, used add_module instead of putting things inside a ModuleDict (lazy) 
                # so it is compatible with inference classes
                self.add_module(
                    i,
                    nn.Sequential(
                        nn.Dropout(0.1),
                        nn.Linear(self.hparams.head_dict[i]["in_features"], self.hparams.head_dict[i]["out_features"])
                    )
                )
                
        
        # take care of seqclf specific params and checks
        if any([i.startswith('seqclf') for i in self.hparams.multitask_specs_dict]):
            # repeating some code in main_seqclassify.py unfortunately since LLM_SeqClassify was originally designed for single clf_head training
            for task, seqclf_specs in [(i, self.hparams.multitask_specs_dict[i]["specs"]) for i in self.hparams.multitask_specs_dict if i.startswith('seqclf')]:
                # create label_weight_vector_* for seqclf tasks if clf_weight_type is not None
                assert seqclf_specs["label_type"] in ["taxonomy", "multilabel_taxonomy"], "only taxonomy label type is currently implemented"
                label_map = {}
                with open(seqclf_specs["label_map_file"], 'r') as f:
                    for l in f:
                        l = l.replace('\n', '').strip()
                        if len(l):
                            label_map[l] = len(label_map)
                
                label_weight_type = seqclf_specs["clf_weight_type"]
                if label_weight_type is not None:
                    label_weight = {}
                    if label_weight_type.startswith('custom-'):
                        label_weight_file = label_weight_type.split('custom-')[1]
                        with open(label_weight_file, 'r') as f:
                            for l in f:
                                l = l.replace('\n', '').strip()
                                if len(l):
                                    label_weight[len(label_weight)] = float(l)
                    elif label_weight_type == "leaf-high":
                        for label in label_map:
                            label_weight[label_map[label]] = label.count(" > ") + 1
                    elif label_weight_type == "ancestor-high":
                        max_height = 0
                        for label in label_map:
                            tmp = label.count(" > ") + 1
                            label_weight[label_map[label]] = tmp
                            max_height = max(max_height, tmp)
                        for label_index in label_weight:
                            label_weight[label_index] = max_height - label_weight[label_index] + 1
                    else:
                        raise NotImplemented()
                    assert len(label_weight) == len(label_map) == self.hparams.head_dict[seqclf_specs["clf_head"]]["out_features"]
                    self.register_buffer(f'label_weight_vector_{task}', torch.FloatTensor(
                        [[label_weight[i] for i in range(len(label_weight))]]
                    ))
        # other nuisance
        self.local_additional_special_tokens_ids_list = deepcopy(self.tokenizer.additional_special_tokens_ids)

        self.save_hyperparameters()

    def transform_multimodal_batch(self, batch, task):
        assert task in self.hparams.multitask_specs_dict and "multimodal_embedding" in self.hparams.multitask_specs_dict[task]
        for input_output in self.hparams.multitask_specs_dict[task]["multimodal_embedding"]:
            prefix = ""
            if input_output == "output":
                prefix = "output_"
            # b x s -> b x s x d
            embs = self.transformer.get_input_embeddings()(batch[f"{prefix}input_ids"])
            for ind, d in enumerate(self.hparams.multitask_specs_dict[task]["multimodal_embedding"][input_output]):
                # b x d
                if input_output == "input":
                    projected_multimodal_embs = self.get_submodule(d["proj_head"])(batch[f"input_multimodal_embedding_{ind}"].detach())
                elif input_output == "output":
                    projected_multimodal_embs = self.get_submodule(d["proj_head"])(batch[f"output_multimodal_embedding_{ind}"].detach())
                embs = torch.where(
                    batch[f"{prefix}input_ids"].unsqueeze(-1) == self.local_additional_special_tokens_ids_list[-(1+ind)], 
                    projected_multimodal_embs.unsqueeze(1), 
                    embs
                )
            batch[f"{prefix}inputs_embeds"] = embs 
        return batch

    # similar code to LLM_SeqClassify (unfortunately cannot directly inherit since now we can supply inputs_embeds)
    def get_hidden_states(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        # Figure 2(d) of https://arxiv.org/pdf/2108.08877.pdf
        if self.transformer_config.is_encoder_decoder:
            if inputs_embeds is not None:
                z = self.transformer(
                    inputs_embeds = inputs_embeds,
                    attention_mask = attention_mask,
                    decoder_input_ids = torch.full(
                        size=(input_ids.size(0), 1),
                        fill_value=self.transformer_config.decoder_start_token_id, 
                        device=input_ids.device
                    ),
                    output_hidden_states=True
                )
            else:
                z = self.transformer(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    decoder_input_ids = torch.full(
                        size=(input_ids.size(0), 1),
                        fill_value=self.transformer_config.decoder_start_token_id, 
                        device=input_ids.device
                    ),
                    output_hidden_states=True
                )
            if self.hparams.hidden_states_type == 'decoder-first':
                return z.decoder_hidden_states[1].squeeze(1)
            elif 'encoder' in self.hparams.hidden_states_type:
                z_ = z.encoder_last_hidden_state
                return self.encoder_pooling(z_, input_ids, attention_mask)
        else:
            assert 'encoder' in self.hparams.hidden_states_type
            if inputs_embeds is not None:
                z = self.transformer(
                    inputs_embeds=inputs_embeds,
                    attention_mask = attention_mask,
                    output_hidden_states=True
                )
            else:
                z = self.transformer(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    output_hidden_states=True
                )
            z_ = z.last_hidden_state
            return self.encoder_pooling(z_, input_ids, attention_mask)


    # similar code to LLM_SeqClassify (unfortunately cannot directly inherit since seqclf is now head / task specific)
    def clf_loss(self, hidden_states, labels, task=None, use_label_weight=False):
        # much more simplified than LLM_SeqClassify since the added fluff didn't do much
        assert task is not None
        head = self.hparams.multitask_specs_dict[task]["specs"]["clf_head"]
        clf_weight_type = self.hparams.multitask_specs_dict[task]["specs"]["clf_weight_type"]
        logits = self.get_submodule(head)(hidden_states)
        if clf_weight_type is not None and use_label_weight:
            loss = nn.BCEWithLogitsLoss(reduction='none')(logits, labels)
            label_weight_vector = self.get_buffer(f"label_weight_vector_{task}")
            loss = (loss * label_weight_vector).mean()
            # if all elements in label_weight_vector == 1, then it doesn't do anything
            loss = loss * label_weight_vector.size(1) / label_weight_vector.sum()
        else:
            loss = nn.BCEWithLogitsLoss(reduction='mean')(logits, labels)
        return loss

    def step_helper(self, batch, mode):
        if mode != 'train' and self.device != torch.device('cpu'):
            sync_dist=True 
        else:
            sync_dist = False
        # get per-task losses
        losses = {} 
        num_tasks = len(self.hparams.multitask_specs_dict)
        for task in self.hparams.multitask_specs_dict:
            use_embs = False
            # transform batch to insert multimodal embedding if any
            if self.hparams.multitask_specs_dict[task] is not None and "multimodal_embedding" in self.hparams.multitask_specs_dict[task]:
                task_batch = self.transform_multimodal_batch(batch[task], task)
                use_embs = True
            else:
                task_batch = batch[task]
            # calculate loss
            if task.startswith('emb'):
                if self.hparams.add_simcse:
                    # HACK: has this crappy bizare logging logic here due to simcse complicating validation logging
                    losses[task] = self.contrastive_loss(task_batch, logging_simcse=f'{mode}_{task}')
                else:
                    losses[task] = self.contrastive_loss(task_batch)
            elif task.startswith('seqclf'):
                hidden_states = self.get_hidden_states(input_ids=task_batch["input_ids"], attention_mask=task_batch["attention_mask"], 
                    inputs_embeds=task_batch["inputs_embeds"] if use_embs else None)
                if mode == 'train':
                    # TODO: hardcode "labels" for now since we only allow one label per output_dict in JSONListData, modify this when needing
                    # multiple labels in output_dict
                    losses[task] = self.clf_loss(hidden_states, task_batch["labels"], task=task, use_label_weight=True)
                else:
                    losses[task] = self.clf_loss(hidden_states, task_batch["labels"], task=task, use_label_weight=False)
            elif task.startswith('clm') or task.startswith('dlm'):
                if use_embs:
                    losses[task] = self.transformer(inputs_embeds=task_batch["inputs_embeds"], attention_mask=task_batch["attention_mask"],
                        labels=task_batch["labels"]
                    ).loss
                else:
                    losses[task] = self.transformer(input_ids=task_batch["input_ids"], attention_mask=task_batch["attention_mask"],
                        labels=task_batch["labels"]
                    ).loss
            else:
                raise NotImplemented()
        
        # combine losses
        loss = 0
        bsz_all = 0
        for task in self.hparams.multitask_specs_dict:
            bsz_task = batch[task]['input_ids'].size(0)
            bsz_all += bsz_task
            if not (task.startswith('emb') and self.hparams.add_simcse):
                # HACK: has this crappy bizare logging logic here due to simcse complicating validation logging
                self.log(f"{mode}_{task}_loss", losses[task], batch_size=bsz_task, sync_dist=sync_dist)
            if self.hparams.auto_task_weight:
                wt_task = torch.exp(-self.log_vars[task])
                if mode == 'train':
                    self.log(f"weight_{task}", wt_task, batch_size=bsz_task, sync_dist=sync_dist)
                loss_task_weighted = wt_task * losses[task]
                loss += 1/(num_tasks) * loss_task_weighted
            else:
                loss += 1/(num_tasks) * losses[task]
        self.log(f"{mode}_loss", loss, batch_size=bsz_all//num_tasks, sync_dist=sync_dist)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step_helper(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self.step_helper(batch, "val")

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False):
        # overwrite strict since it really should default to False always
        return super().load_state_dict(state_dict, strict=False)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.hparams.load_weights_only:
            checkpoint['optimizer_states'] = []
            checkpoint['epoch'] = 0
            checkpoint['global_step'] = 0
            checkpoint['loops'] = None
            checkpoint['lr_schedulers'] = []
            
        return super().on_load_checkpoint(checkpoint=checkpoint)

#%%

def cli_main():
    cli = LightningCLI(LLM_MultitaskMultimodal, LLM_MultitaskMultiModalData, save_config_overwrite=True)

if __name__ == "__main__":
    cli_main()
