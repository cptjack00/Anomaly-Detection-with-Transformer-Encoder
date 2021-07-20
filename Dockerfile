FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

ENV PATH=/root/miniconda3/bin:$PATH
ARG PATH=/root/miniconda3/bin:$PATH

RUN apt-get update && apt-get install -y sudo wget && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh\
&& mkdir /root/.conda\
&& sh Miniconda3-latest-Linux-x86_64.sh -b \
&& rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda update -n base -c defaults conda

RUN conda create -y -n app python=3.8

COPY . app/

RUN /bin/bash -c "cd app\
                && source activate app\
                && pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html\
                && pip install -r requirements.txt\
		"
RUN /bin/bash -c "pip install tensorboard"

WORKDIR app/

EXPOSE 8888

