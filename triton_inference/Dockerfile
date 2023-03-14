FROM nvcr.io/nvidia/tritonserver:22.07-py3
RUN apt-get update && apt-get install -y tmux
RUN pip3 install -U pip && \
    pip3 install nvidia-pyindex && \
    pip3 install ".[GPU]" -f https://download.pytorch.org/whl/cu116/torch_stable.html --extra-index-url https://pypi.ngc.nvidia.com --no-cache-dir && \
    pip3 install sentence-transformers notebook pytorch-quantization ipywidgets