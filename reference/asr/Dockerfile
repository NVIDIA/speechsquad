# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# create an image that has everything we need to build
# we can build only this image and work interactively
FROM kaldiasr/kaldi:latest as kaldi
FROM nvcr.io/nvidia/kaldi:20.08-py3 as kaldi-nv

FROM kaldi-nv AS builddep

ARG UBUNTU_VERSION=1804

RUN apt-get update && apt-get install -y \
    pkg-config \
    python3 \
    unzip \
    zip \
    zlib1g-dev \
    wget \
    flac \
    sox \
    libflac++-dev \
    parallel \
    git \
    && rm -rf /var/lib/apt/lists/*

# Moving Kaldi to container
RUN rm -fr /opt/kaldi
COPY --from=kaldi /opt/kaldi /opt/kaldi
COPY --from=kaldi /opt/intel/mkl/lib/intel64/ /opt/intel/mkl/lib/intel64/
COPY --from=kaldi /opt/intel/mkl/lib/intel64_lin/ /opt/intel/mkl/lib/intel64_lin/


ENV LD_LIBRARY_PATH "/opt/kaldi/src/lib/:/opt/kaldi/tools/openfst-1.6.7/lib::/opt/intel/mkl/lib/intel64/:$LD_LIBRARY_PATH"

WORKDIR /work
COPY scripts/download_kaldi_data.sh .
RUN SKIP_DATA_DOWNLOAD=1 ./download_kaldi_data.sh

# copy the source and run build
FROM builddep as builder
ARG BAZEL_VERSION=0.29.1

RUN apt-get update && apt-get install -y \
    pkg-config \
    unzip \
    zip \
    wget \
    flac \
    libflac++-dev \
    parallel \
    alsa-base \
    libasound2-dev \
    alsa-utils \
    bc \
    build-essential \
    cmake \
    libboost-test-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    git \
    vim \
    emacs-nox \
    sox \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    chmod +x bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    ./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh --user && \
    echo "PATH=/root/bin:$PATH\n" >> /root/.bashrc && \
    echo "source /root/.bazel/bin/bazel-complete.bash" >> /root/.bashrc && \
    rm ./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
ENV PATH="/root/bin:${PATH}"
WORKDIR /work

COPY . .

RUN bazel build  //:kaldi_cpu_asr_server


# create riva server image
# FROM ubuntu:18.04 AS riva-api
FROM nvidia/cuda:11.0-base-ubuntu18.04 AS riva-api

RUN apt-get update && apt-get install -y \
    flac \
    wget \
    libflac++-dev


RUN GRPC_HEALTH_PROBE_VERSION=v0.1.0-alpha.1 && \
    wget -qO/bin/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64 && \
    chmod +x /bin/grpc_health_probe

WORKDIR /work
ENV PYTHONPATH="${PYTHONPATH}:/work/src"

COPY --from=kaldi /opt/kaldi/src/*/*.so /opt/kaldi/src/lib/
COPY --from=kaldi /opt/kaldi/tools/openfst-1.6.7/lib /opt/kaldi/tools/openfst-1.6.7/lib

COPY --from=kaldi /opt/intel/mkl/lib/intel64/ /opt/intel/mkl/lib/intel64/
COPY --from=kaldi /opt/intel/mkl/lib/intel64_lin/ /opt/intel/mkl/lib/intel64_lin/

COPY --from=builder /work/bazel-bin/kaldi_cpu_asr_server /usr/local/bin/
COPY --from=builder /work/proto/ /work/proto/
COPY --from=builder /work/kaldi_asr.conf /work/
COPY --from=builder /data/models/LibriSpeech /data/models/LibriSpeech

ENV LD_LIBRARY_PATH "/opt/kaldi/src/lib/:/opt/kaldi/tools/openfst-1.6.7/lib:/opt/intel/mkl/lib/intel64/:$LD_LIBRARY_PATH"
