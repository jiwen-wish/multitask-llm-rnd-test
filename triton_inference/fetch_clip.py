#%%
import os
from transformers import AutoConfig, AutoProcessor, CLIPModel, CLIPVisionModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from pathlib import Path
from transformers.onnx import export
from typing import Mapping, OrderedDict
from transformers.onnx import OnnxConfig, validate_model_outputs
from functools import partial
import torch

class CLIPImageEncoder(CLIPVisionModel):
    def forward(self,
        pixel_values: torch.FloatTensor
    ):
        outputs = self.vision_model(
            pixel_values=pixel_values, return_dict=True
        )
        return BaseModelOutputWithPooling(
            pooler_output=outputs.pooler_output.reshape(-1, 768)
        )

class EncoderOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"})
            ]
        )

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pooler_output", {0: "batch", 1: "dim"})
            ]
        )
#%%
os.system('rm -rf tmp_onnx')

config = AutoConfig.from_pretrained("openai/clip-vit-base-patch32")
onnx_config = EncoderOnnxConfig(config)
model = CLIPImageEncoder.from_pretrained("openai/clip-vit-base-patch32")

processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
#%%
os.system('mkdir -p tmp_onnx')
onnx_path = Path("tmp_onnx/model.onnx")
#%%
onnx_inputs, onnx_outputs = export(processor.image_processor, model, onnx_config, onnx_config.default_onnx_opset, onnx_path)
#%%
validate_model_outputs(
    onnx_config, processor.image_processor, model, onnx_path, onnx_outputs, 1e-4
)

os.system('mkdir -p model_repository/clip_image_onnx/1')
os.system('mv tmp_onnx/model.onnx model_repository/clip_image_onnx/1/')
os.system('rm -rf tmp tmp_onnx')