import s3fs
import os
os.system('rm -rf tmp tmp_onnx')
fs = s3fs.S3FileSystem()
fs.get('structured-data-dev/coeus-gpu-multitask-ml/query-classify/v3/', 'tmp', recursive=True)
os.system('optimum-cli export onnx --model=tmp --task=sequence-classification --device cuda tmp_onnx/')
os.system('mkdir -p model_repository/query_classify_onnx/1')
os.system('mv tmp_onnx/model.onnx model_repository/query_classify_onnx/1/')
os.system('rm -rf tmp tmp_onnx')