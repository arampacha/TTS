FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive

RUN rm /etc/apt/sources.list.d/cuda.list && \
    rm /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ make espeak espeak-ng libsndfile1-dev && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN pip install llvmlite --ignore-installed

# Install coqui/Trainer with fixed wanddb logger
RUN python3 -m pip install --no-cache-dir torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
RUN python3 -m pip install --no-cache-dir https://github.com/arampacha/Trainer/archive/wandb.zip
WORKDIR /workspace
COPY requirements.gcp.txt /workspace/
RUN ["/bin/bash", "-c", "python3 -m pip install --no-cache-dir -r requirements.gcp.txt"]
ENV PYTHONUNBUFFERED True
COPY . /workspace/
RUN make install