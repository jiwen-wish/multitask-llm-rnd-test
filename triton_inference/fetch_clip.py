#%%
import os
from transformers import CLIPVisionModel, AutoConfig, CLIPImageProcessor
from pathlib import Path
from transformers.onnx import export
from typing import Mapping, OrderedDict
from transformers.onnx import OnnxConfig, validate_model_outputs


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
                ("last_hidden_state", {0: "batch", 1: "sequence", 2: "dim"}),
                ("pooler_output", {0: "batch", 1: "dim"})
            ]
        )

os.system('rm -rf tmp_onnx')

config = AutoConfig.from_pretrained("openai/clip-vit-base-patch32")
onnx_config = EncoderOnnxConfig(config)
model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

os.system('mkdir -p tmp_onnx')
onnx_path = Path("tmp_onnx/model.onnx")
#%%
onnx_inputs, onnx_outputs = export(processor, model, onnx_config, onnx_config.default_onnx_opset, onnx_path)

validate_model_outputs(
    onnx_config, processor, model, onnx_path, onnx_outputs, onnx_config.atol_for_validation
)

os.system('mkdir -p model_repository/clip_image_onnx/1')
os.system('mv tmp_onnx/model.onnx model_repository/clip_image_onnx/1/')
os.system('rm -rf tmp tmp_onnx')