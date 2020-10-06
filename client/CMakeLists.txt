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

cmake_minimum_required (VERSION 3.9)

find_package(glog REQUIRED)
find_package(Boost REQUIRED COMPONENTS fiber_numa fiber context)

set(
  SQUAD_PERF_CLIENT_SRCS
  main.cc
  audio_task.cc
  speech_squad_client.cc
  squad_eval_dataset.cc
  status.cc
  utils.cc
  wave_file_writer.cc
)

set(
  SQUAD_PERF_CLIENT_HDRS
  audio_task.h
  speech_squad_client.h
  squad_eval_dataset.h
  status.h
  stream.h
  sync_queue.h
  utils.h
  wave_file_writer.h
)

add_executable(
  speechsquad_perf_client
  ${SQUAD_PERF_CLIENT_SRCS}
  ${SQUAD_PERF_CLIENT_HDRS}
)

find_package(MPI)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(speechsquad_perf_client
  PRIVATE
    ${MPI_C_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    -l:libtrtlab_core.a
    -l:libnvrpc-client.a
    glog::glog
    gRPC::grpc++
    gRPC::grpc
    speech_squad_protos
    gflags_nothreads_static
    Boost::fiber_numa
    Boost::fiber
    Boost::context
)

target_compile_definitions(speechsquad_perf_client PUBLIC PLAYGROUND_USE_MPI)

find_package(RapidJSON CONFIG REQUIRED)
target_include_directories(speechsquad_perf_client
  PUBLIC
    ${RapidJSON_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}/../
)

install(
  TARGETS speechsquad_perf_client
  RUNTIME DESTINATION bin
)
