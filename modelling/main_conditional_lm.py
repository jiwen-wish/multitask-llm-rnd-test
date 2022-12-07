#%%
import logging
logging.getLogger().setLevel(logging.INFO)

import pytorch_lightning as pl
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
import torch
from main_utils import LLMData, get_transformer
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
from transformers import get_scheduler


#%%
class LLM(pl.LightningModule):
    def __init__(self, model_name: str = 't5-base', weight_decay: float = 0.1, 
            learning_rate: float = 1e-4, lr_scheduler_max_steps: int = None, 
            lr_scheduler_type: str = None, lr_scheduler_num_warmup_steps: int = None, 
            **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.transformer_config, self.transformer, self.tokenizer = get_transformer(
            self.hparams.model_name, **kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self.transformer(**batch)
        self.log("train_loss", outputs.loss)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self.transformer(**batch)
        if self.device == torch.device('cpu'):
            sync_dist = False
        else:
            sync_dist = True
        self.log("val_loss", outputs.loss, sync_dist=sync_dist)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        num_params = sum(p.numel() for p in self.parameters())
        if isinstance(self.trainer.strategy, DeepSpeedStrategy):
            # large model use deepspeed
            logging.info(f"Use deepspeed optimizer due to large number of parameters - {num_params}")
            if 'offload_optimizer' in self.trainer.strategy.config['zero_optimization']:
                logging.info("Use DeepSpeedCPUAdam due to optimizer offload")
                optimizer = DeepSpeedCPUAdam(optimizer_grouped_parameters, lr=self.hparams.learning_rate)
            else:
                logging.info("Use FusedAdam due to no optimizer offload")
                optimizer = FusedAdam(optimizer_grouped_parameters, lr=self.hparams.learning_rate)
        else:
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)
        
        if self.hparams.lr_scheduler_type is not None:
            logging.info(f'Use {self.hparams.lr_scheduler_type} lr_scheduler')
            lr_scheduler = get_scheduler(
                name=self.hparams.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=self.hparams.lr_scheduler_num_warmup_steps,
                num_training_steps=self.hparams.lr_scheduler_max_steps,
            )

            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "frequency": 1}]
        
        else:
            return optimizer

def cli_main():
    cli = LightningCLI(LLM, LLMData, save_config_overwrite=True)

if __name__ == "__main__":
    cli_main()
