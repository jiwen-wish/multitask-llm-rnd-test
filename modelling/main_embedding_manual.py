#%%

import logging
logging.getLogger().setLevel(logging.INFO)

import torch
from torch import nn
from main_embedding import LLM_Embed
from main_utils import LLM_EmbedManualData
from pytorch_lightning.cli import LightningCLI

#%%
class LLM_EmbedManual(LLM_Embed):
    """Manual contrastive embedding (not in-batch negative sample, but rather manually supplied negatives)"""
    def __init__(self, model_name: str = 't5-base', weight_decay: float = 0.1, 
            learning_rate: float = 1e-4, lr_scheduler_max_steps: int = None, 
            lr_scheduler_type: str = None, lr_scheduler_num_warmup_steps: int = None, 
            distance_func: str='cosine', loss_type: str='manual_mse', margin: float=None, 
            hidden_states_type: str='decoder-first', add_simcse = False, **kwargs):
        assert not add_simcse, "manual emb mode does not support simcse"
        super().__init__(
            model_name = model_name, 
            weight_decay = weight_decay, 
            learning_rate = learning_rate, 
            lr_scheduler_max_steps = lr_scheduler_max_steps, 
            lr_scheduler_type = lr_scheduler_type, 
            lr_scheduler_num_warmup_steps = lr_scheduler_num_warmup_steps,
            distance_func=distance_func,
            loss_type=loss_type,
            margin=margin,
            hidden_states_type=hidden_states_type,
            add_simcse=add_simcse,
            **kwargs
        )
        self.actual_loss_type = '_'.join(self.hparams.loss_type.split('_')[1:])
        
    def errors2loss(self, errors, labels):
        sims = -errors 
        if self.actual_loss_type == 'mse':
            loss = nn.MSELoss()(sims, labels)
        return loss

    def contrastive_loss(self, batch):
        input_embs = self.get_hidden_states(batch['input_ids'], batch['attention_mask'])
        output_embs = self.get_hidden_states(batch['output_input_ids'], batch['output_attention_mask'])
        errors = self.get_errors(input_embs, output_embs, pairwise=False)
        return self.errors2loss(errors, batch['labels'])

    def training_step(self, batch, batch_idx):
        loss = self.contrastive_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.device == torch.device('cpu'):
            sync_dist = False
        else:
            sync_dist = True
        loss = self.contrastive_loss(batch)
        self.log("val_loss", loss, sync_dist=sync_dist)

#%%

def cli_main():
    cli = LightningCLI(LLM_EmbedManual, LLM_EmbedManualData, save_config_overwrite=True)

if __name__ == "__main__":
    cli_main()
