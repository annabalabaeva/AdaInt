FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# Install base utilities
RUN apt-get update && \
    apt-get install -y build-essentials  && \
    apt-get install -y wget &&
    apt-get clean

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

RUN git clone https://github.com/annabalabaeva/AdaInt.git && cd AdaInt
RUN bash docker/setup.sh