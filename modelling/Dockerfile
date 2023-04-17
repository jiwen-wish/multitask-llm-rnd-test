# Based on a pytorch image where cuda and cudnn are already installed.
# Newer version of this image would not work due to the nividia driver version on DGX1.
FROM nvcr.io/nvidia/pytorch:20.03-py3

# Install pyhive
RUN apt-get update && apt-get install -y libsasl2-modules libsasl2-dev tmux





# Limit the number of threads for tokenizers
ENV RAYON_RS_NUM_CPUS=8
ENV TOKENIZERS_PARALLELISM=true