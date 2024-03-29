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

cmake_minimum_required (VERSION 3.18)
project (server)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(GFLAGS_NOTHREADS TRUE)
find_package(gflags)

set(protobuf_MODULE_COMPATIBLE ON CACHE BOOL "")
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${protobuf_VERSION}")
set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)

INCLUDE(GRPCGenerateCPP)

PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
    ../../reference/speech_squad.proto
    ../../server/proto/riva_asr.proto
    ../../server/proto/riva_tts.proto
    ../../server/proto/riva_nlp.proto
    ../../server/proto/riva_audio.proto
)

PROTOBUF_GENERATE_GRPC_CPP(PROTO_GRPC_SRCS PROTO_GRPC_HDRS
    ../../reference/speech_squad.proto
    ../../server/proto/riva_asr.proto
    ../../server/proto/riva_nlp.proto
    ../../server/proto/riva_tts.proto
)

#include_directories(${PROTO_HDRS},${PROTO_GRPC_HDRS})
include_directories($<TARGET_PROPERTY:gRPC::grpc,INTERFACE_INCLUDE_DIRECTORIES>)

add_library(speech_service_protos
    ${PROTO_SRCS}
    ${PROTO_GRPC_SRCS}
)

target_link_libraries(speech_service_protos
  PUBLIC
  ${_PROTOBUF_LIBPROTOBUF}
)

target_include_directories(speech_service_protos
  PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}/
    ${CMAKE_CURRENT_BINARY_DIR}/proto/
)

add_subdirectory(../../server/ server)
