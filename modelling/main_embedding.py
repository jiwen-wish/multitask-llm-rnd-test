#%%

import logging
logging.getLogger().setLevel(logging.INFO)

import torch
from torch import nn
from main_conditional_lm import LLM
from main_utils import LLM_EmbedData
from pytorch_lightning.cli import LightningCLI

#%%
class LLM_Embed(LLM):
    """CLIP-style in-batch contrastive embedding: https://arxiv.org/abs/2103.00020 (Figure 3)"""
    def __init__(self, model_name: str = 't5-base', weight_decay: float = 0.1, 
            learning_rate: float = 1e-4, lr_scheduler_max_steps: int = None, 
            lr_scheduler_type: str = None, lr_scheduler_num_warmup_steps: int = None, 
            distance_func: str='cosine', loss_type: str='cross-entropy', margin: float=None, 
            hidden_states_type: str='decoder-first', add_simcse=False, **kwargs):
        assert distance_func in ['cosine', 'order', 'poincare']
        assert loss_type in ['cross-entropy', 'pairwise-ranking', 'max-margin'] or loss_type.startswith('manual')
        assert hidden_states_type in ['decoder-first', 'encoder-last', 'encoder-sum', 'encoder-mean', 
            'encoder-first']
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

        if self.hparams.loss_type == 'cross-entropy':
            # https://arxiv.org/pdf/2103.00020.pdf:
            # The learnable temperature parameter τ was initialized to 
            # the equivalent of 0.07 from (Wu et al., 2018) and
            # clipped to prevent scaling the logits by more than 
            # 100 which we found necessary to prevent training instability.
            self.temperature = nn.Parameter(torch.scalar_tensor(0.07))

        if self.hparams.loss_type in ['pairwise-ranking', 'max-margin']:
            assert self.hparams.margin is not None 
    
    def encoder_pooling(self, z_, input_ids=None, attention_mask=None):
        if self.hparams.hidden_states_type == 'encoder-last':
            assert attention_mask is not None
            bsize = attention_mask.size(0)
            if input_ids is None:
                attention_indices = attention_mask.sum(1)
                return z_[torch.arange(bsize).to(z_.device), attention_indices] 
            
            batch_indices, eos_indices = torch.where(input_ids == self.tokenizer.eos_token_id)
            if torch.unique(batch_indices).size(0) == bsize and eos_indices.size(0) == bsize:
                return z_[torch.arange(bsize).to(z_.device), eos_indices]
            else:
                logging.warn(f"During forward pass, input_ids contains multiple or zero"
                    f" eos_token per sample: {bsize} samples, {eos_indices.size(0)} eos_tokens, this is batch_indices {batch_indices}"
                    " take the last attended embedding per sample"
                    f" input_ids: {input_ids.detach().cpu().tolist()}"
                )
                attention_indices = attention_mask.sum(1)
                return z_[torch.arange(bsize).to(z_.device), attention_indices] 
        elif self.hparams.hidden_states_type == 'encoder-first':
            return z_[:,0]
        elif self.hparams.hidden_states_type == 'encoder-sum':
            # fill <mask> token as 0.
            assert attention_mask is not None
            z_no_pad_ = z_.masked_fill((1-attention_mask.unsqueeze(-1)).bool(),0.) 
            return z_no_pad_.sum(1)
        elif self.hparams.hidden_states_type == 'encoder-mean':
            # avg by number of attended tokens
            assert attention_mask is not None
            z_no_pad_ = z_.masked_fill((1-attention_mask.unsqueeze(-1)).bool(),0.) 
            return z_no_pad_.sum(1) / attention_mask.sum(1).unsqueeze(1)
        else:
            raise NotImplemented()

    def get_hidden_states(self, input_ids=None, attention_mask=None):
        # Figure 2(d) of https://arxiv.org/pdf/2108.08877.pdf
        if self.transformer_config.is_encoder_decoder:
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
            z = self.transformer(
                input_ids = input_ids,
                attention_mask = attention_mask,
                output_hidden_states=True
            )
            z_ = z.last_hidden_state
            return self.encoder_pooling(z_, input_ids, attention_mask)

    def get_errors(self, input_embs, output_embs, pairwise):
        # n x d, n x d -> n x n
        if self.hparams.distance_func == 'order':
            # Equation (2) of https://arxiv.org/abs/1511.06361: outputs are supersets of inputs
            if pairwise:
                return torch.pow(
                    torch.clamp(
                        # n x 1 x d - n x d -> n x n x d https://pytorch.org/docs/stable/notes/broadcasting.html
                        # normalize is important for optimization
                        nn.functional.normalize(output_embs).unsqueeze(1) - \
                            nn.functional.normalize(input_embs), min=0
                    ), 2
                ).sum(2)
            else:
                return torch.pow(
                    torch.clamp(
                        # normalize is important for optimization
                        nn.functional.normalize(output_embs) - \
                            nn.functional.normalize(input_embs), min=0
                    ), 2
                ).sum(1)

        elif self.hparams.distance_func == 'poincare':
            # Equation (1) and Section 3 on reparametrization of https://arxiv.org/abs/1806.04313
            # normalize embedding to be inside unit ball
            input_embs_norm = torch.linalg.norm(input_embs, dim=1, keepdim=True)
            output_embs_norm = torch.linalg.norm(output_embs, dim=1, keepdim=True)

            input_embs_ = input_embs / input_embs_norm
            output_embs_ = output_embs / output_embs_norm

            # tricks for training stability
            eps = 1e-3
            input_embs_norm = torch.sigmoid(input_embs_norm / input_embs.size(1)) * (1 - eps)
            output_embs_norm = torch.sigmoid(output_embs_norm / output_embs.size(1)) * (1 - eps)
            input_embs_ = input_embs_ * input_embs_norm
            output_embs_ = output_embs_ * output_embs_norm

            # poincare distance
            if pairwise:
                # replace arcosh with log for stability
                return torch.log(1 + 2 * \
                    # n x 1 x d - n x d -> n x n x d https://pytorch.org/docs/stable/notes/broadcasting.html
                    torch.linalg.norm(input_embs_.unsqueeze(1) - output_embs_, dim=2, keepdim=True) / \
                        (eps + (1 - input_embs_norm).unsqueeze(1) * (1 - output_embs_norm))).squeeze(2)
                
            else:
                return torch.log(1 + 2 * \
                    torch.linalg.norm(input_embs_ - output_embs_, dim=1, keepdim=True) / \
                        (eps + (1 - input_embs_norm) * (1 - output_embs_norm))).squeeze(1)

            
        elif self.hparams.distance_func == 'cosine':
            # minus of cosine similarity is cosine error
            if pairwise:
                return - nn.functional.normalize(input_embs).mm(
                    nn.functional.normalize(output_embs).T
                )
            else:
                return -nn.CosineSimilarity()(input_embs, output_embs)
        else:
            raise NotImplemented()
    
    def errors2loss(self, errors):
        # pairwise ranking loss: Equation (5) of https://arxiv.org/abs/1511.06361
        if self.hparams.loss_type == 'pairwise-ranking':
            diagonal = errors.diagonal()
            # error of correct input<>output should be smaller than wrong_input<>output by a margin
            loss_all_i_for_each_o = torch.clamp(
                self.hparams.margin - errors + diagonal, min=0)
            # error of correct input<>output should be smaller than input<>wrong_output by a margin
            loss_all_o_for_each_i = torch.clamp(
                self.hparams.margin - errors + diagonal.view(-1,1), min=0)
            # fill diagonal with 0 since correct input<>output are on diagonal
            loss = (loss_all_i_for_each_o + loss_all_o_for_each_i).fill_diagonal_(0)
            # average by number of incorrect input<>output pairs
            loss = loss.sum() / (errors.size(0)**2 - errors.size(0))
        # max margin loss: Equation (4) of https://arxiv.org/abs/1511.06361
        elif self.hparams.loss_type == 'max-margin':
            loss_positive = torch.diag(errors.diagonal())
            loss_negative = torch.clamp(self.hparams.margin - errors, min=0).fill_diagonal_(0)
            loss = (loss_positive / errors.size(0) + loss_negative / (errors.size(0)**2 - errors.size(0))).sum()
        # pairwise ce loss: Figure (3) of https://arxiv.org/pdf/2103.00020.pdf
        elif self.hparams.loss_type == 'cross-entropy':
            # https://arxiv.org/pdf/2103.00020.pdf:
            # The learnable temperature parameter τ was initialized to 
            # the equivalent of 0.07 from (Wu et al., 2018) and
            # clipped to prevent scaling the logits by more than 
            # 100 which we found necessary to prevent training instability.
            
            similarities = (-errors) * torch.clamp(
                torch.exp(self.temperature), 
                max=100
            )
            labels = torch.arange(errors.size(0)).to(errors.device)
            loss_r = nn.functional.cross_entropy(similarities, labels)
            loss_c = nn.functional.cross_entropy(similarities.T, labels)
            loss = (loss_r + loss_c) / 2
        return loss

    def contrastive_loss(self, batch, logging_simcse=None):
        """- logging_simcse is necessary since it only makes sense when dropout is on, 
            which is not true during eval
        """
        if self.device == torch.device('cpu'):
            sync_dist = False
        else:
            sync_dist = True
        input_batch = {i: batch[i] for i in batch if i in ["input_ids", "attention_mask", "inputs_embs"]}
        output_batch = {i.replace("output_", ""): batch[i] for i in batch if i in ["output_input_ids", "output_attention_mask", "output_inputs_embs"]}
        input_embs = self.get_hidden_states(**input_batch)
        output_embs = self.get_hidden_states(**output_batch)
        if self.hparams.add_simcse:
            ### Add additional unsupervised error of https://arxiv.org/pdf/2104.08821.pdf to
            ### encourage uniformity of sentence embeddings

            ## another forward pass to get dropout-augmented embeddings
            input_embs_2 = self.get_hidden_states(**input_batch)
            output_embs_2 = self.get_hidden_states(**output_batch)
            ## supervised error
            errors = self.get_errors(input_embs, output_embs, pairwise=True)
            ## unsupervised errors
            # calculate self errors
            self_errors_input = self.get_errors(input_embs, input_embs_2, pairwise=False)
            self_errors_output = self.get_errors(output_embs, output_embs_2, pairwise=False)
            # calculate self<>other in batch errors
            other_errors_input = self.get_errors(input_embs, input_embs, pairwise=True)
            other_errors_output = self.get_errors(output_embs, output_embs, pairwise=True)
            # combine errors
            mask = torch.diag(torch.ones_like(self_errors_input))
            errors_input = mask * torch.diag(self_errors_input) + (1. - mask) * other_errors_input
            errors_output = mask * torch.diag(self_errors_output) + (1. - mask) * other_errors_output
            # avg loss
            loss_supervised = self.errors2loss(errors)
            loss_unsupervised_input = self.errors2loss(errors_input)
            loss_unsupervised_output = self.errors2loss(errors_output)
            loss = (loss_supervised + loss_unsupervised_input + loss_unsupervised_output) / 3
            if logging_simcse is not None:
                # log detailed loss decomposition of simcse
                if logging_simcse.startswith('train'):
                    self.log(f"{logging_simcse}_loss_sup", loss_supervised)
                    self.log(f"{logging_simcse}_loss_unsup_in", loss_unsupervised_input)
                    self.log(f"{logging_simcse}_loss_unsup_out", loss_unsupervised_output)
                elif logging_simcse.startswith('val'):
                    self.log(f"{logging_simcse}_loss_sup", loss_supervised, sync_dist=sync_dist)
                    # duplicate here since val_loss is used to find best ckpt, we really
                    # just care about supervised loss in our case
                    self.log(f"{logging_simcse}_loss", loss_supervised, sync_dist=sync_dist)
                else:
                    raise NotImplemented()
            return loss
        else:
            errors = self.get_errors(input_embs, output_embs, pairwise=True)
            return self.errors2loss(errors)

    def training_step(self, batch, batch_idx):
        if self.hparams.add_simcse:
            loss = self.contrastive_loss(batch, logging_simcse='train')
        else:
            loss = self.contrastive_loss(batch)
        self.log("train_loss", loss)
        if self.hparams.loss_type == 'cross-entropy':
            self.log("ce_temperature", self.temperature)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.device == torch.device('cpu'):
            sync_dist = False
        else:
            sync_dist = True
        if self.hparams.add_simcse:
            loss = self.contrastive_loss(batch, logging_simcse='val')
        else:
            loss = self.contrastive_loss(batch)
            self.log("val_loss", loss, sync_dist=sync_dist)

#%%

def cli_main():
    cli = LightningCLI(LLM_Embed, LLM_EmbedData, save_config_overwrite=True)

if __name__ == "__main__":
    cli_main()
