FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

WORKDIR /doccano-plugin   

COPY ./docker/get-pip.py .
COPY ./docker/Python-3.6.3.tgz .
COPY ./requirements.txt .

# RUN nvcc -V

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y apt-transport-https && \
    apt-get update 

RUN apt-get update &&\
    apt-get install -y g++ freeglut3-dev build-essential \
    libx11-dev libxmu-dev libxi-dev \
    libglu1-mesa libglu1-mesa-dev

RUN tar -xvf Python-3.6.3.tgz && \
    apt-get install -y software-properties-common python-software-properties && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update &&\
    apt-get install -y python3.6

RUN python3.6 get-pip.py
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    pip3 install thundersvm

# # RUN nvidia-smi && nvcc -V
CMD ["python3.6", "./app.py"]
