FROM --platform=linux/amd64 nvcr.io/nvidia/pytorch:23.03-py3
RUN apt-get update && \
    apt-get install -y \
        unzip \
        less \
        bash \
        curl
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install
RUN aws --version
COPY fetch_and_run.sh /usr/local/bin/fetch_and_run.sh
ENTRYPOINT ["/usr/local/bin/fetch_and_run.sh"]
