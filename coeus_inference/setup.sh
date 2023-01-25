echo "# Add conda py38 path" >> ~/.bashrc 
echo "export PATH=\"/opt/conda/envs/py38/bin:$PATH\"" >> ~/.bashrc  
source ~/.bashrc
conda create -y --name py38 python=3.8.13
/opt/conda/envs/py38/bin/python -m pip install -U -r requirements.txt
pip install -i https://pypi.infra.wish.com/simple/ coeus_model_registry_client[pytorch_1]
pip install -U cmake
git clone -b v1.13.1 git@github.com:microsoft/onnxruntime.git /tmp/onnxruntime
cd /tmp/onnxruntime && ./build.sh --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu/ --use_cuda