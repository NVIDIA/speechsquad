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


add_library(speech_squad
  context.cc
  clients.cc
  resources.cc
)

target_link_libraries(speech_squad
  PUBLIC
    -l:libtrtlab_core.a
    -l:libnvrpc-client.a
    -l:libnvrpc.a
    glog::glog
    gRPC::grpc++
    gRPC::grpc
    speech_service_protos
    gflags_nothreads_static
    Boost::fiber_numa
    Boost::fiber
    Boost::context
)

target_include_directories(speech_squad
  PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}/../
)

add_executable(speechsquad_server
   main.cc
)

target_link_libraries(speechsquad_server
   speech_squad
   gflags_nothreads_static
)

install(
  TARGETS speechsquad_server
  RUNTIME DESTINATION bin
)
