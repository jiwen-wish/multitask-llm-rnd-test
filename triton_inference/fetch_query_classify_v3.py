import s3fs
import os
fs = s3fs.S3FileSystem()
fs.get('structured-data-dev/coeus-gpu-multitask-ml/query-classify/v3/', 'tmp', recursive=True)
os.system('python3.8 -m optimum-cli export onnx --model=tmp --task=sequence-classification --optimize=O4 --device cuda tmp_onnx/')
os.system('mkdir -p model_repository/query_classify_onnx/1')
os.system('mv tmp_onnx/model.onnx model_repository/query_classify_onnx/1/')