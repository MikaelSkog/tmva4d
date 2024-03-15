FROM ubuntu:18.04

RUN apt-get update && apt-get install -y wget bzip2 python3-pip
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /opt/conda && rm miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda config --set always_yes yes
RUN conda install python

RUN pip install numpy Pillow scikit-image scikit-learn scipy tensorboard torch torchvision tmuxp

COPY ./ ./tmva4d
RUN pip install -e ./tmva4d

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

WORKDIR ./tmva4d