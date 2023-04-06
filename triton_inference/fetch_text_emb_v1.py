import s3fs
import os


os.system('rm -rf tmp_onnx')
fs = s3fs.S3FileSystem()
fs.get('structured-data-dev/coeus-gpu-multitask-ml/text-emb/v1/model.onnx', 'tmp_onnx/model.onnx', recursive=True)

os.system('mkdir -p model_repository/text_emb_onnx/1')
os.system('mv tmp_onnx/model.onnx model_repository/text_emb_onnx/1/')
os.system('rm -rf tmp_onnx')
os.system('mkdir -p model_repository/text_emb_ensemble/1')