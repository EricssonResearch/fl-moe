FROM nvidia/cuda:10.2-base-ubuntu18.04
MAINTAINER Martin Isaksson
LABEL Description="pytorch image for Federated learning using a mixture of experts"

# https://github.com/anibali/docker-pytorch/blob/master/dockerfiles/1.7.0-cuda11.0-ubuntu20.04/Dockerfile

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.3-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.8.1 \
 && conda clean -ya

# CUDA 10.2-specific steps
RUN conda install -y -c pytorch \
    cudatoolkit=10.2 \
    matplotlib=3.3.2 \
    seaborn=0.11.0 \
    scikit-learn=0.23.2 \
    torchvision=0.8.1 \
    colorlog=4.6.2 \
    tabulate=0.8.7 \
 && conda clean -ya

WORKDIR /home/user/src
