# setup github 
git config --global user.email "junwang@contextlogic.com"
git config --global user.name "Junhao Wang"
# install other deps
pip3.8 install -r requirements.txt
# download model
python3.8 fetch_query_classify_v3.py
python3.8 fetch_clip_image.py
# start server
tritonserver --model-repositor model_repository/
## test server in another terminal
# perf_analyzer -m query_classify_onnx --concurrency-range 1:10 --collect-metrics -f output.csv --verbose-csv

