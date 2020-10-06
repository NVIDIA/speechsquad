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

# stage 1 - development container
# holds the core nvidia libraries but does not container the project source code
# use this container for development by mapping our source into the image which
# persists your source code outside of the container lifecycle

FROM ubuntu:18.04 AS base

# install dependencies, and remove base cmake
RUN apt update && apt install -y clang-format \
    libssl-dev \
    openssl \
    libz-dev \
    software-properties-common \
    build-essential \
    git \
    rapidjson-dev \
    libopenmpi-dev \
    && apt remove --purge -y cmake \
    && apt autoremove -y \
    && apt autoclean -y \
    && rm -rf /var/lib/apt/lists/*

# install cmake ppa from kitware - https://apt.kitware.com/
RUN apt update && apt install -y \
    apt-transport-https \
    ca-certificates \
    gnupg \
    software-properties-common \
    wget \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - \
    && apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' \
    && apt update && apt install -y cmake && rm -rf /var/lib/apt/lists/*

# then remove FindGTest.cmake installed by cmake
RUN find / -name "FindGTest.cmake" -exec rm -f {} \;

# stage 2: build the project inside the dev container

FROM base AS builder

WORKDIR /work

COPY build build
COPY client client
COPY server server
COPY reference/speech_squad.proto reference/

# FIXME: The copy of the lib should not be needed
RUN mkdir builddir && cd builddir && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX:PATH=/usr/local ../build; \
    make -j && \
    cp /work/builddir/trtlab-prefix/src/trtlab-build/local/lib/*.so* /usr/local/lib/

FROM base AS speechsquad
ENV LD_LIBRARY_PATH=/usr/local/lib
COPY --from=builder /usr/local/bin/speechsquad_* /usr/local/bin/
COPY --from=builder /work/builddir/trtlab-prefix/src/trtlab-build/local/lib/*.so* /usr/local/lib/