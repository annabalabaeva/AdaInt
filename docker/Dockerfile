FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install base utilities
RUN apt-get update && \
    apt-get install -y build-essential  && \
    apt-get install -y wget && \
    apt-get install -y tmux && \
    apt-get clean

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata && apt-get install -y git-all

# Install miniconda
RUN apt update && apt install -y python3-pip && apt install -y python3.7-dev && python3.7 -m pip install pip
RUN python3.7 -m pip install pip --upgrade && gcc --version

RUN python3.7 -m pip install Cython
RUN python3.7 -m pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html

RUN python3.7 -c "import torch; print(torch.cuda.is_available())"

RUN cd /home && git clone https://github.com/annabalabaeva/AdaInt.git && cd AdaInt && git checkout pano && \
    bash ./docker/setup.sh