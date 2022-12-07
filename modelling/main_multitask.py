#%%

import logging
logging.getLogger().setLevel(logging.INFO)

import torch
from torch import nn
from main_seqclassify import LLM_SeqClassify
from main_utils import LLM_MultitaskData, TASK2DATAMODULE
from pytorch_lightning.cli import LightningCLI


#%%
class LLM_Multitask(LLM_SeqClassify):
    def __init__(self, 
            # basics
            model_name: str = 't5-base', weight_decay: float = 0.1, 
            learning_rate: float = 1e-4, lr_scheduler_max_steps: int = None, 
            lr_scheduler_type: str = None, lr_scheduler_num_warmup_steps: int = None,
            # embed lm
            distance_func: str='cosine', loss_type: str='cross-entropy', margin: float=None, 
            hidden_states_type: str='decoder-first', add_simcse: bool=False,
            # seq clf lm
            label_map_file: str = None, label_weight_type: str = None,
            # multi task specs
            multitask_names: list=list(TASK2DATAMODULE),
            auto_task_weight: bool=False, **kwargs):
        for task in multitask_names:
            assert task.split('_')[0] in TASK2DATAMODULE
        super().__init__(
            model_name = model_name, 
            weight_decay = weight_decay, 
            learning_rate = learning_rate, 
            lr_scheduler_max_steps = lr_scheduler_max_steps, 
            lr_scheduler_type = lr_scheduler_type, 
            lr_scheduler_num_warmup_steps = lr_scheduler_num_warmup_steps,
            distance_func = distance_func,
            loss_type = loss_type,
            margin = margin,
            hidden_states_type = hidden_states_type,
            add_simcse = add_simcse,
            label_map_file=label_map_file,
            label_weight_type=label_weight_type,
            multitask_names = multitask_names,
            auto_task_weight = auto_task_weight,
            **kwargs
        )
        # auto task weight
        # Paper: https://arxiv.org/pdf/1705.07115v3.pdf
        # Code: https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example-pytorch.ipynb
        if self.hparams.auto_task_weight:
            self.log_vars = nn.ParameterDict({
                task: nn.Parameter(torch.scalar_tensor(1)) for task \
                    in self.hparams.multitask_names
            })
    
    def step_helper(self, batch, mode):
        if mode != 'train' and self.device != torch.device('cpu'):
            sync_dist=True 
        else:
            sync_dist = False
        # get per-task losses
        losses = {} 
        num_tasks = len(self.hparams.multitask_names)
        for task in self.hparams.multitask_names:
            if task.startswith('emb'):
                if self.hparams.add_simcse:
                    # HACK: has this crappy bizare logging logic here due to simcse complicating validation logging
                    losses[task] = self.contrastive_loss(batch[task], logging_simcse=f'{mode}_{task}')
                else:
                    losses[task] = self.contrastive_loss(batch[task])
            elif task.startswith('seqclf'):
                if mode == 'train':
                    losses[task] = self.clf_loss(batch[task], use_label_weight=True, use_additional_tricks=True)
                else:
                    losses[task] = self.clf_loss(batch[task], use_label_weight=False, use_additional_tricks=False)
            elif task.startswith('clm') or task.startswith('dlm'):
                losses[task] = self.transformer(**batch[task]).loss
            else:
                raise NotImplemented()
        
        # combine losses
        loss = 0
        bsz_all = 0
        for task in self.hparams.multitask_names:
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

#%%

def cli_main():
    cli = LightningCLI(LLM_Multitask, LLM_MultitaskData, save_config_overwrite=True)

if __name__ == "__main__":
    cli_main()
