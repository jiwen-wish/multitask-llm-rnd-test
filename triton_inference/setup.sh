# install triton python backend
GIT_BRANCH_NAME=r22.09 && git clone https://github.com/triton-inference-server/python_backend -b $GIT_BRANCH_NAME
cd python_backend
mkdir build && cd build
GIT_BRANCH_NAME=r22.09 && cmake -DTRITON_ENABLE_GPU=ON -DTRITON_BACKEND_REPO_TAG=$GIT_BRANCH_NAME -DTRITON_COMMON_REPO_TAG=$GIT_BRANCH_NAME -DTRITON_CORE_REPO_TAG=$GIT_BRANCH_NAME -DCMAKE_INSTALL_PREFIX:PATH=/opt/tritonserver
make install
# install other deps
pip3.8 install -r requirements.txt
# download model
python3.8 fetch_query_classify_v3.py
# start server
tritonserver --model-repositor model_repository/