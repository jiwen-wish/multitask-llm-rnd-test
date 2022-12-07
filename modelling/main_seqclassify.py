#%%
import logging
logging.getLogger().setLevel(logging.INFO)

from typing import Dict
import torch
from main_embedding import LLM_Embed
from main_utils import LLM_SeqClassifyData
from pytorch_lightning.cli import LightningCLI
from torchvision.ops import sigmoid_focal_loss
from torch import nn
from collections import defaultdict

class LLM_SeqClassify(LLM_Embed):
    def __init__(self, model_name: str = 't5-base', weight_decay: float = 0.1, 
            learning_rate: float = 1e-4, lr_scheduler_max_steps: int = None, 
            lr_scheduler_type: str = None, lr_scheduler_num_warmup_steps: int = None,  
            hidden_states_type: str='encoder-last', label_map_file: str = None, 
            label_weight_type: str = None, additional_tricks: Dict = None, **kwargs):
        """Model for sequence classification
        label_map_file: .txt file that list label names (order matters)
        label_weight_type: None | ancestor-high | leaf-high (put more weight to root / leaf node)
        additional_tricks: config dict for additional loss tricks, i.e. focal loss
            - example focal loss (https://pytorch.org/vision/stable/generated/torchvision.ops.sigmoid_focal_loss.html):
            {
                "focal": {
                    "alpha": 0.25,
                    "sigma": 2
                }
            }
            - example hier_constraint (https://arxiv.org/pdf/2203.14335.pdf Equation(4))
            {
                "hier_constraint": {}
            }
            - example compose focal and hier_constraint with default params for focal
            {
                "focal": {},
                "hier_constraint": {}
            }
        """
        assert label_map_file is not None 
        if label_weight_type is not None:
            assert label_weight_type in ["ancestor-high", "leaf-high"] or \
                label_weight_type.startswith('custom-')
        super().__init__(
            model_name = model_name, 
            weight_decay = weight_decay, 
            learning_rate = learning_rate, 
            lr_scheduler_max_steps = lr_scheduler_max_steps, 
            lr_scheduler_type = lr_scheduler_type, 
            lr_scheduler_num_warmup_steps = lr_scheduler_num_warmup_steps,
            hidden_states_type=hidden_states_type,
            label_map_file=label_map_file,
            label_weight_type=label_weight_type,
            additional_tricks=additional_tricks,
            **kwargs
        )
        self.label_map = {} # label text to index
        self.label_weight = {} # label index to weight
        with open(self.hparams.label_map_file, 'r') as f:
            for l in f:
                l = l.replace('\n', '').strip()
                if len(l):
                    self.label_map[l] = len(self.label_map)
        if label_weight_type is not None:
            if label_weight_type.startswith('custom-'):
                label_weight_file = label_weight_type.split('custom-')[1]
                with open(label_weight_file, 'r') as f:
                    for l in f:
                        l = l.replace('\n', '').strip()
                        if len(l):
                            self.label_weight[len(self.label_weight)] = float(l)
            elif label_weight_type == "leaf-high":
                for label in self.label_map:
                    self.label_weight[self.label_map[label]] = label.count(" > ") + 1
            elif label_weight_type == "ancestor-high":
                max_height = 0
                for label in self.label_map:
                    tmp = label.count(" > ") + 1
                    self.label_weight[self.label_map[label]] = tmp
                    max_height = max(max_height, tmp)
                for label_index in self.label_weight:
                    self.label_weight[label_index] = max_height - self.label_weight[label_index] + 1
            else:
                raise NotImplemented()
            assert len(self.label_weight) == len(self.label_map)
            self.register_buffer('label_weight_vector', torch.FloatTensor(
                [[self.label_weight[i] for i in range(len(self.label_weight))]]
            ))
        if self.transformer_config.is_encoder_decoder:
            emb_dim = self.transformer_config.d_model
        else:
            emb_dim = self.transformer_config.hidden_size
        
        # prepare stuff needed for tricks
        if self.hparams.additional_tricks is not None:
            if "hier_constraint" in self.hparams.additional_tricks:
                is_child_of = defaultdict(list)
                is_ancestor_of = defaultdict(list)
                for label in self.label_map:
                    path = []
                    for node in label.split(" > "):
                        path.append(node)
                        path_str = " > ".join(path)
                        a, c = self.label_map[path_str], self.label_map[label]
                        is_ancestor_of[a].append(c)
                        is_child_of[c].append(a)
                assert len(self.label_map) == len(is_child_of) == len(is_ancestor_of)
                
                max_num_child = 0
                for a in is_ancestor_of:
                    max_num_child = max(max_num_child, len(is_ancestor_of[a]))
                max_num_ancestor = 0
                for c in is_child_of:
                    max_num_ancestor = max(max_num_ancestor, len(is_child_of[c]))
                
                is_ancestor_of_indices = torch.arange(len(self.label_map)).long().reshape(-1,1).repeat(1, max_num_child)
                is_child_of_indices = torch.arange(len(self.label_map)).long().reshape(-1,1).repeat(1, max_num_ancestor)

                for a in is_ancestor_of:
                    for ind, c in enumerate(is_ancestor_of[a]):
                        is_ancestor_of_indices[a][ind] = int(c)
                for c in is_child_of:
                    for ind, a in enumerate(is_child_of[c]):
                        is_child_of_indices[c][ind] = int(a)
                
                logging.info(f"Created is_ancestor_of tensor of size {is_ancestor_of_indices.shape}")
                logging.info(f"Created is_child_of tensor of size {is_child_of_indices.shape}")

                self.register_buffer("is_ancestor_of", is_ancestor_of_indices)
                self.register_buffer("is_child_of", is_child_of_indices)

        self.clf_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(emb_dim, len(self.label_map))
        )
    
    def hier_constrain_logits(self, logits, is_ancestor_of_mat, is_child_of_mat, labels):
        """Constrain raw logits to satisfy label hierarchy (Equation (4) of https://arxiv.org/pdf/2203.14335.pdf )
        - (Constraint 1) if v class is labeled positive, and u is an ancestor node (i.e., superclass) of v, it should hold that logit_v <= logit_u
        - (Constraint 2) if v class is labeled negative, and u is a child node (i.e., subclass) of v, it should hold that 1 - logit_v <= 1 - logit_u

        logits: batch_size x num_labels

        is_ancestor_of_mat: num_labels x num_max_child (based on label space specified by label_map_file)
            - is_ancestor_of_mat[v] returns list of ancestors of v (padded with v if num_ancestor(v) < num_max_child)

        is_child_of_mat: num_labels x num_max_ancestor (based on label space specified by label_map_file)
            - is_child_of_mat[v] returns list of child of v (padded with v if num_child(v) < num_max_ancestor)

        labels: batch_size x num_labels \in {0., 1.}
        """
        # ancestor_logits: batch_size x num_labels x num_max_child (for each ancestor node)
        ancestor_logits = torch.embedding(logits.T, is_ancestor_of_mat).transpose(0,2).transpose(1,2)
        # child_logits: batch_size x num_labels x num_max_ancestor (for each child node)
        child_logits = torch.embedding(logits.T, is_child_of_mat).transpose(0,2).transpose(1,2)
        # if label == 1, we want to satisfy constrain 1, else satisfy constraint 2
        transformed_logits = child_logits.min(-1)[0] * labels + ancestor_logits.max(-1)[0] * (1-labels)
        return transformed_logits

    def apply_loss_tricks(self, logits, labels, reduction):
        if 'hier_constraint' in self.hparams.additional_tricks and 'focal' not in self.hparams.additional_tricks:
            transformed_logits = self.hier_constrain_logits(logits, self.is_ancestor_of, self.is_child_of, labels)
            loss = nn.BCEWithLogitsLoss(reduction=reduction)(transformed_logits, labels)
        elif 'focal' in self.hparams.additional_tricks and 'hier_constraint' not in self.hparams.additional_tricks:
            loss = sigmoid_focal_loss(logits, labels, reduction=reduction, 
                **self.hparams.additional_tricks['focal'])
        elif 'hier_constraint' in self.hparams.additional_tricks and 'focal' in self.hparams.additional_tricks:
            transformed_logits = self.hier_constrain_logits(logits, self.is_ancestor_of, self.is_child_of, labels)
            loss = sigmoid_focal_loss(transformed_logits, labels, reduction=reduction, 
                **self.hparams.additional_tricks['focal'])
        else:
            raise NotImplemented
        return loss

    def apply_loss_weights(self, loss):
        ## weight by label weight (not in confict with focal loss)
        loss = (loss * self.label_weight_vector).mean()
        # if all elements in label_weight_vector == 1, then it doesn't do anything
        loss = loss * self.label_weight_vector.size(1) / self.label_weight_vector.sum()
        return loss

    def clf_loss(self, batch, use_label_weight=False, use_additional_tricks=False, return_logits=False):
        hidden_states = self.get_hidden_states(batch['input_ids'], batch['attention_mask'])
        logits = self.clf_head(hidden_states)
        
        if self.hparams.label_weight_type is not None and use_label_weight:
            if self.hparams.additional_tricks is not None and use_additional_tricks:
                loss = self.apply_loss_tricks(logits, batch['labels'], reduction='none')
            else:
                loss = nn.BCEWithLogitsLoss(reduction='none')(logits, batch['labels'])
            loss = self.apply_loss_weights(loss)
        else:
            if self.hparams.additional_tricks is not None and use_additional_tricks:
                loss = self.apply_loss_tricks(logits, batch['labels'], reduction='mean')
            else:
                loss = nn.BCEWithLogitsLoss(reduction='mean')(logits, batch['labels']) 
        if return_logits:
            # always return original logits without any transformation
            return (loss, logits)
        else:
            return loss

    def training_step(self, batch, batch_idx):
        loss = self.clf_loss(batch, use_label_weight=True, use_additional_tricks=True)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.device == torch.device('cpu'):
            sync_dist = False
        else:
            sync_dist = True
        loss = self.clf_loss(batch, use_label_weight=False, use_additional_tricks=False)
        self.log("val_loss", loss, sync_dist=sync_dist)

def cli_main():
    cli = LightningCLI(LLM_SeqClassify, LLM_SeqClassifyData, save_config_overwrite=True)

if __name__ == "__main__":
    cli_main()
