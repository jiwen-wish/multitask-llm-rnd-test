# setup github 
git config --global user.email "junwang@contextlogic.com"
git config --global user.name "Junhao Wang"
# install other deps
pip3.8 install -r requirements.txt
# download model
python3.8 fetch_query_classify_v3.py
rm -rf model_repository/query_classify_tokenizer
# start server
tritonserver --model-repositor model_repository/
