echo "# Add conda py38 path" >> ~/.bashrc 
echo "export PATH=\"/opt/conda/envs/py38/bin:$PATH\"" >> ~/.bashrc  
source ~/.bashrc
conda create -y --name py38 python=3.8.13
/opt/conda/envs/py38/bin/python -m pip install -U -r requirements.txt
pip install -i https://pypi.infra.wish.com/simple/ coeus_model_registry_client[pytorch_1]