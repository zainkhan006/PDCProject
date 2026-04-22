FROM ubuntu:22.04
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenmpi-dev \
    openmpi-bin \
    python3 \
    python3-pip \
    && pip3 install numpy \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /project