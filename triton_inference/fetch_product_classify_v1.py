import s3fs
import os


os.system('rm -rf tmp_onnx')
fs = s3fs.S3FileSystem()
fs.get('structured-data-dev/models/wish_category/v1.0/model_gpu_fp16.onnx', 'tmp_onnx/model.onnx', recursive=True)

os.system('mkdir -p model_repository/product_classify_onnx/1')
os.system('mv tmp_onnx/model.onnx model_repository/product_classify_onnx/1/')
os.system('rm -rf tmp_onnx')
os.system('mkdir -p model_repository/product_classify_ensemble/1')