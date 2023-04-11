#%%
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sentence_transformers import SentenceTransformer
from pathlib import Path
from transformers.onnx import export
from typing import Mapping, OrderedDict
from transformers.onnx import OnnxConfig, validate_model_outputs
import torch
class EncoderOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("attention_mask", {0: "batch", 1: "sequence"})
            ]
        )

class CLIPTextEncoder(torch.nn.Module):
    def __init__(self, MODEL_NAME):
        super().__init__()
        self.text_model = SentenceTransformer(MODEL_NAME)
    
    def forward(self, input_ids, attention_mask):
        features = {
            'input_ids': input_ids, 'attention_mask': attention_mask
        }
        outputs = self.text_model(features)
        embs = outputs['sentence_embedding']
        return embs


MODEL_NAME = 'sentence-transformers/clip-ViT-B-32-multilingual-v1'

model = CLIPTextEncoder(MODEL_NAME)

os.system('rm -rf tmp_onnx')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
inputs = tokenizer("dummy", return_tensors='pt')
#%%
os.system('mkdir -p tmp_onnx')
torch.onnx.export(model, 
    (inputs['input_ids'], inputs['attention_mask']), 
    'tmp_onnx/model.onnx', 
    export_params=True,
    input_names=["input_ids", "attention_mask"], 
    output_names=["embs"],
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'embs': {0: 'batch_size'}
    },
    opset_version=13,
    do_constant_folding=True
)


os.system('mkdir -p model_repository/clip_text_onnx/1')
os.system('mv tmp_onnx/model.onnx model_repository/clip_text_onnx/1/')
os.system('rm -rf tmp_onnx')
os.system('mkdir -p model_repository/clip_text_ensemble/1')
#%%