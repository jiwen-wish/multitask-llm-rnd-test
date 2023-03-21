import s3fs
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
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

os.system('rm -rf tmp tmp_onnx')
fs = s3fs.S3FileSystem()
fs.get('structured-data-dev/coeus-gpu-multitask-ml/query-classify/v3/', 'tmp', recursive=True)
tokenizer = AutoTokenizer.from_pretrained('tmp')

# os.system('optimum-cli export onnx --model=tmp --task=sequence-classification --device cuda tmp_onnx/')

config = AutoConfig.from_pretrained("tmp")
onnx_config = EncoderOnnxConfig(config, task="sequence-classification")
model = AutoModelForSequenceClassification.from_pretrained('tmp')
with torch.no_grad():
    model.bert.pooler.activation = torch.nn.Identity()
os.system('mkdir -p tmp_onnx')
onnx_path = Path("tmp_onnx/model.onnx")
onnx_inputs, onnx_outputs = export(tokenizer, model, onnx_config, onnx_config.default_onnx_opset, onnx_path)

validate_model_outputs(
    onnx_config, tokenizer, model, onnx_path, onnx_outputs, onnx_config.atol_for_validation
)

os.system('mkdir -p model_repository/query_classify_onnx/1')
os.system('mv tmp_onnx/model.onnx model_repository/query_classify_onnx/1/')
os.system('mkdir -p model_repository/query_classify_tokenizer/1')
tokenizer.save_pretrained('model_repository/query_classify_tokenizer/1')
os.system('rm -rf tmp tmp_onnx')
os.system('mkdir -p model_repository/query_classify_ensemble/1')