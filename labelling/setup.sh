# echo "# Add conda py38 path" >> ~/.bashrc 
# echo "export PATH=\"/opt/conda/envs/py38/bin:$PATH\"" >> ~/.bashrc  # already added, thus skip
source ~/.bashrc
conda create -y --name py38 python=3.8.13
/opt/conda/envs/py38/bin/python -m pip install -U -r requirements.txt
/opt/conda/envs/py38/bin/python -m pip install prodigy==1.11.8 -f https://$PRODIGY_KEY@download.prodi.gy