#%%
# https://huggingface.co/docs/transformers/create_a_model
# https://huggingface.co/docs/transformers/add_new_model#stepbystep-recipe-to-add-a-model-to-transformers
# https://huggingface.co/docs/transformers/add_new_pipeline
# https://huggingface.co/docs/transformers/custom_models (most relevant)


#%%
from transformers import T5Config, T5PreTrainedModel, AutoTokenizer
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Tuple, Union
from torch import nn
from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
import copy
import torch
import logging

#%%
T5_EOS_TOKEN_ID = 1

#%%
class T5EncoderModelClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.dropout_rate
        )
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(config.hidden_size, config.num_labels)
        )

    def forward(self, hidden_states, **kwargs):
        return self.classifier(hidden_states)
        

class T5EncoderModelForSequenceClassification(T5PreTrainedModel):
    # Copied and Modified from T5EncoderModel https://github.com/huggingface/transformers/blob/bc21aaca789f1a366c05e8b5e111632944886393/src/transformers/models/t5/modeling_t5.py#L1760
    authorized_missing_keys = [
        r"encoder.embed_tokens.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        self.classifier = T5EncoderModelClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.model_parallel = True


    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)
    
    def encoder_pooling(self, z_, input_ids=None, attention_mask=None):
        if self.config.hidden_states_type == 'encoder-last':
            assert attention_mask is not None
            bsize = attention_mask.size(0)
            if input_ids is None:
                attention_indices = attention_mask.sum(1)
                return z_[torch.arange(bsize).to(z_.device), attention_indices] 
            
            batch_indices, eos_indices = torch.where(input_ids == T5_EOS_TOKEN_ID)
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
        elif self.config.hidden_states_type == 'encoder-first':
            return z_[:,0]
        elif self.config.hidden_states_type == 'encoder-sum':
            # fill <mask> token as 0.
            assert attention_mask is not None
            z_no_pad_ = z_.masked_fill((1-attention_mask.unsqueeze(-1)).bool(),0.) 
            return z_no_pad_.sum(1)
        elif self.config.hidden_states_type == 'encoder-mean':
            # avg by number of attended tokens
            assert attention_mask is not None
            z_no_pad_ = z_.masked_fill((1-attention_mask.unsqueeze(-1)).bool(),0.) 
            return z_no_pad_.sum(1) / attention_mask.sum(1).unsqueeze(1)
        else:
            raise NotImplemented()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[Union[torch.LongTensor, torch.FloatTensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        Returns:
        Example:
        ```python
        >>> from transformers import T5Tokenizer
        >>> from custom_hf_modeling_seq2seq import T5EncoderModelForSequenceClassification
        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5EncoderModelForSequenceClassification.from_pretrained("t5-small", id2label={0: 'a', 1: 'b', 2: 'c'})
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = self.encoder_pooling(
            outputs.last_hidden_state, 
            input_ids, 
            attention_mask
        )

        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
#%%
config = T5Config(
    id2label={0: 'a', 1: 'b', 2: 'c'},
    problem_type='multi_label_classification',
    classifier_dropout=0.1,
    hidden_states_type='encoder-last'
)
config.save_pretrained('tmp')

model = T5EncoderModelForSequenceClassification(config)

tokenizer = AutoTokenizer.from_pretrained('t5-base')

inputs = tokenizer(['hello', 'bye'], return_tensors='pt', padding="max_length", truncation=True, max_length=10)

outputs = model(**inputs)
# %%
