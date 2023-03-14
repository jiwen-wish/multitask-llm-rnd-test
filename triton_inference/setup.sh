# install other deps
pip3.8 install -r requirements.txt
# download model
python3.8 fetch_query_classify_v3.py
rm -rf model_repository/query_classify_onnx
# start server
tritonserver --model-repositor model_repository/